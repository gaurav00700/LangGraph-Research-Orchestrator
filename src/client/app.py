"""
This module provides the interactive web interface for the multi-agent system.
It manages user sessions, real-time message streaming from the FastAPI backend, 
dynamic plan updates in the sidebar, and file download artifacts.
"""
import streamlit as st
import requests
import uuid
import os
import json

# --- Configuration ---
API_URL = "http://localhost:8000/chat"
# Correctly resolve output directory relative to server
# Correctly resolve output directory relative to server
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
HISTORY_URL = "http://localhost:8000/history"
UPLOAD_URL = "http://localhost:8000/upload"


st.set_page_config(page_title="VIBE Planning Agent", layout="wide")

st.title("Multi-Agent Research Chatbot")
st.markdown("""This specialist agent creates a **technical plan** for your goal and executes both server-side and client-side actions.""")

# --- Session State Management ---
# We use st.session_state to maintain the conversation thread and UI state 
# (like the active plan step) across Streamlit's frequent re-renders.

# --- Session State ---
if "session_id" not in st.session_state:
    # URL Query param -> Session State
    query_params = st.query_params
    url_session_id = query_params.get("session_id", None)
    
    if url_session_id:
        st.session_state.session_id = url_session_id
        st.session_state.history_loaded = False # Trigger load
    else:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history_loaded = True # New session, empty history
        st.query_params["session_id"] = st.session_state.session_id

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Load History from Backend ---
if not st.session_state.get("history_loaded", False):
    try:
        import requests
        resp = requests.get(f"{HISTORY_URL}/{st.session_state.session_id}")
        if resp.status_code == 200:
            data = resp.json()
            # Append backend messages to session state if unique
            backend_msgs = data.get("messages", [])
            st.session_state.messages = backend_msgs
            st.toast("Restored Conversation History")
        st.session_state.history_loaded = True
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        st.session_state.history_loaded = True # Stop trying

if "current_plan" not in st.session_state:
    st.session_state.current_plan = ""
if "active_step_index" not in st.session_state:
    st.session_state.active_step_index = 0
if "downloads" not in st.session_state:
    st.session_state.downloads = []

# --- Sidebar ---
with st.sidebar:
    st.header("Agent Status")
    st.info(f"Session ID: `{st.session_state.session_id}`")
    
    if st.button("Reset Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.query_params["session_id"] = st.session_state.session_id
        st.session_state.current_plan = ""
        st.session_state.active_step_index = -1
        st.rerun()

    resumption_id = st.text_input("Resume Session ID", value=st.session_state.session_id)
    if st.button("Resume Session", use_container_width=True):
        try:
            # Check headers or status to verify existance? 
            # Ideally we just set the session_id and trigger a reload which fetches history
            st.session_state.session_id = resumption_id
            st.query_params["session_id"] = resumption_id
            st.session_state.history_loaded = False
            st.rerun()
        except Exception as e:
            st.error(f"Error resuming: {e}")

    # st.markdown("---")
    # Placeholder for Stop Button (Only active during generation)
    sidebar_stop_placeholder = st.empty()
    sidebar_stop_placeholder.button("🛑 Stop Generation", disabled=True, key="stop_gen_idle", use_container_width=True)

    st.markdown("---")
    st.subheader("📁 Knowledge Upload")
    uploaded_file = st.file_uploader("Upload PDF/TXT/MD to Knowledge Base", type=["pdf", "txt", "md"])
    if uploaded_file is not None:
        if st.button("Ingest File in Vector DB", use_container_width=True):
            with st.spinner("Ingesting..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    resp = requests.post(UPLOAD_URL, files=files)
                    if resp.status_code == 200:
                        st.success(f"Indexed {uploaded_file.name}!")
                        st.toast("File added to Vector Store!")
                    else:
                        st.error(f"Failed: {resp.text}")
                except Exception as e:
                    st.error(f"Upload Error: {e}")

    st.markdown("---")
    st.subheader("📋 Execution Plan")
    # Create a dynamic container for the plan that can be updated during streaming
    plan_container = st.empty()
    
    st.subheader("⬇️ Downloads")
    downloads_container = st.empty()

    def render_sidebar():
        with downloads_container.container():
            if st.session_state.downloads:
                for i, f in enumerate(st.session_state.downloads):
                    fname = f.get("filename", "file")
                    fpath = os.path.join(OUTPUT_DIR, fname)
                    if os.path.exists(fpath):
                         with open(fpath, "rb") as file:
                            st.download_button(f"⬇️ {fname}", data=file, file_name=fname, key=f"dl_{i}_{fname}")
            else:
                st.caption("No artifacts yet.")

    render_sidebar()

    with plan_container.container():
        if st.session_state.current_plan:
            plan_steps = st.session_state.current_plan.strip().split("\n")
            for i, step in enumerate(plan_steps):
                clean_step = step.lstrip("- ").strip()
                if i == st.session_state.active_step_index:
                    st.markdown(f"**{i+1}. 🎯 {clean_step}**")
                elif i < st.session_state.active_step_index:
                    st.markdown(f"{i+1}. ✅ {clean_step}")
                else:
                    st.markdown(f"{i+1}. 📝 {clean_step}")
        else:
            st.write("Waiting for goal...")

# --- Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Render status/trace if available (reconstruction from history)
        if message.get("trace"):
            with st.status(message.get("status_label", "Execution Complete"), state="complete", expanded=False):
                for t in message["trace"]:
                    st.write(t)
            with st.expander("🔍 Execution Trace", expanded=False):
                for t in message["trace"]:
                    st.markdown(f"*{t}*")
        
        st.markdown(message["content"])

if prompt := st.chat_input("What is your goal?"):
    # Clear old plan when new message is entered
    st.session_state.current_plan = ""
    st.session_state.active_step_index = -1
    st.session_state.downloads = []
    
    # Force UI update
    plan_container.empty()
    downloads_container.empty()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Placeholders for live updates
        status_container = st.status("Agent is working...", expanded=True)
        trace_expander = st.expander("🔍 Execution Trace", expanded=False)
        
        # Activate Sidebar Stop Button (Enabled)
        stop_generation = False
        if sidebar_stop_placeholder.button("🛑 Stop Generation", key="stop_gen_active", use_container_width=True, disabled=False):
            st.toast("Generation Stopped!")
            stop_generation = True
            # st.rerun() # Rerun will clear the current stream, so we handle stop_generation flag instead
            
        message_placeholder = st.empty()
        full_response = ""
        last_active_node = None
        shown_step_headers = set()
        token_buffer = "" 
        trace_messages = []
        
        try:
            payload = {
                "message": prompt,
                "thread_id": st.session_state.session_id
            }
            
            # Start streaming request
            response = requests.post(API_URL, json=payload, stream=True)
            response.raise_for_status()
            # Custom SSE Event Consumer:
            # We iterate over the streaming response lines from the FastAPI server.
            # Each line is expected to be a valid NDJSON object with a 'type' field.
            for line in response.iter_lines():
                if stop_generation: 
                    break
                if line:
                    event = json.loads(line.decode("utf-8"))
                    e_type = event.get("type")
                
                    # 1. Tracing
                    if e_type == "trace":
                        node_name = event.get("node", "Agent")
                        icon = event.get("icon", "🔄")
                        
                        # Filter out internal LangGraph nodes
                        internal_nodes = ["RunnableSequence", "RunnableLambda", "Unnamed", "LangGraph", "agent", "call_model", "should_continue", "tools", "Prompt"]
                        if node_name not in internal_nodes:
                            trace_msg = f"{icon} **{node_name}** is working..."
                            status_container.write(trace_msg)
                            trace_messages.append(trace_msg)
                            # Update trace expander
                            with trace_expander:
                                for msg in trace_messages:
                                    st.markdown(f"*{msg}*")
                    
                    # 2. Plan Generation
                    elif e_type == "plan":
                        plan_content = event["content"]
                        # Filter out JSON output - only show if it's a formatted plan (starts with -)
                        if plan_content and not plan_content.strip().startswith("{"):
                            st.session_state.current_plan = plan_content
                            st.session_state.active_step_index = -1
                            status_container.write("📋 **Plan Generated**:")
                            status_container.markdown(plan_content)
                            
                            # Update sidebar plan dynamically
                            with plan_container.container():
                                plan_steps = plan_content.strip().split("\n")
                                for i, step in enumerate(plan_steps):
                                    clean_step = step.lstrip("- ").strip()
                                    st.markdown(f"{i+1}. 📝 {clean_step}")
                    
                    # 3. Active Step Highlighting & Grouping
                    elif e_type == "active_step":
                        new_step_index = event["index"]
                        step_desc = event.get("description", "")
                        st.session_state.active_step_index = new_step_index
                        node_icon = event.get("icon", "🎯")
                        
                        # Update sidebar plan to show active step
                        if st.session_state.current_plan:
                            with plan_container.container():
                                plan_steps = st.session_state.current_plan.strip().split("\n")
                                for i, step in enumerate(plan_steps):
                                    clean_step = step.lstrip("- ").strip()
                                    if i == new_step_index:
                                        st.markdown(f"**{i+1}. 🎯 {clean_step}**")
                                    elif i < new_step_index:
                                        st.markdown(f"{i+1}. ✅ {clean_step}")
                                    else:
                                        st.markdown(f"{i+1}. 📝 {clean_step}")
                    
                    # 4. Node Results
                    elif e_type == "node_result":
                        node_name = event.get("node", "Agent")
                        content = event.get("content", "")
                        
                        # Filter out internal nodes and JSON content
                        internal_nodes = ["RunnableSequence", "RunnableLambda", "Unnamed", "LangGraph", "agent", "call_model", "should_continue", "tools", "Prompt", "planner", "supervisor"]
                        is_json = content.strip().startswith("{") if content else False
                        
                        if content and content.strip() and node_name not in internal_nodes and not is_json:
                            # Only show in status container, not in main response (to avoid duplicates)
                            with status_container:
                                with st.expander(f"✅ Result from **{node_name}**", expanded=False):
                                    st.markdown(content)
                    
                    # 5. Final User-facing Tokens
                    elif e_type == "token":
                        token_content = event["content"]
                        
                        # Filter out JSON tokens (from planner/supervisor) - only if it looks like complete JSON
                        is_json_token = False
                        if token_content:
                            stripped = token_content.strip()
                            # Only filter if it's clearly JSON (starts with { and contains common JSON keys)
                            if stripped.startswith("{") and any(key in stripped for key in ['"plan_steps"', '"next_worker"', '"initial_worker"']):
                                is_json_token = True
                        
                        if token_content and not is_json_token:
                            # Flush buffer first if needed
                            if token_buffer:
                                full_response += token_buffer
                                token_buffer = ""
                            # Always append tokens to full response
                            full_response += token_content
                            message_placeholder.markdown(full_response + "▌")
                    
                    # 6. Server Actions (Observability Events)
                    elif e_type == "server_action":
                        action_type = event.get("action")
                        data = event.get("data", {})
                        
                        if action_type == "show_download":
                            fname = event.get("filename")
                            if not any(d.get("filename") == fname for d in st.session_state.downloads):
                                st.session_state.downloads.append({"filename": fname})
                                render_sidebar()
                                st.toast(f"File Ready: {fname}", icon="💾")
                        
                        elif action_type == "outline_update":
                            header = data.get("header", "")
                            if header:
                                status_container.markdown(f"📝 **{header}**")
                                
                    # 7. Final actions
                    elif e_type == "final":
                        # Final flush of buffer if any
                        if token_buffer:
                            full_response += "\n" + token_buffer
                            token_buffer = ""
                            
                        message_placeholder.markdown(full_response + "▌")
                        
                        # Update sidebar to show all steps as completed
                        if st.session_state.current_plan:
                            with plan_container.container():
                                plan_steps = st.session_state.current_plan.strip().split("\n")
                                for i, step in enumerate(plan_steps):
                                    st.markdown(f"{i+1}. ✅ {step.lstrip('- ').strip()}")
                                            
            # End of response stream processing
            status_container.update(label="Planning & Execution Complete", state="complete", expanded=False)
            
            # Finalize assistant message in history
            if full_response:
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "trace": trace_messages,
                    "status_label": "Planning & Execution Complete"
                })
            else:
                 # If no tokens were specifically streamed (e.g. only node results), show a summary
                 final_msg = "Task completed based on the steps above."
                 message_placeholder.markdown(final_msg)
                 st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_msg,
                    "trace": trace_messages,
                    "status_label": "Planning & Execution Complete"
                 })

        except Exception as e:
            st.error(f"Error: {str(e)}")
            status_container.update(label="Execution Failed", state="error")
        finally:
            # Return to Inactive State
            sidebar_stop_placeholder.button("🛑 Stop Generation", disabled=True, key="stop_gen_final", use_container_width=True)
