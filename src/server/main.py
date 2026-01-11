"""
This module hosts the REST API that connects the Streamlit/CLI clients to 
the LangGraph multi-agent system. It provides endpoints for streaming chat 
responses, retrieval of session history, and file ingestion into the 
Vector Store.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
import uuid
import json
import traceback

import sys
import os

# Add src to sys.path to ensure 'agent' package is found
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.append(src_dir)
root_dir = os.path.abspath(os.path.join(src_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from contextlib import asynccontextmanager
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent.graph import workflow, DB_PATH
from agent.tools import index_content
import aiosqlite
from fastapi import UploadFile, File
import io
import pypdf


# --- Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the FastAPI application lifespan.
    
    Startup:
    1. Connects to the SQLite database using aiosqlite.
    2. Monkeypatches the `is_alive` attribute onto the connection for LangGraph compatibility.
    3. Initializes the AsyncSqliteSaver checkpointer.
    4. Compiles the LangGraph workflow with the checkpointer.
    
    Shutdown:
    - Closes the DB connection automatically (via context manager).
    """
    # Startup: Open DB and Compile Agent
    print(f"Server Startup. Connecting to Checkpoint DB: {DB_PATH}")
    async with aiosqlite.connect(DB_PATH) as conn:
        # Monkeypatch is_alive for LangGraph compatibility
        # LangGraph AsyncSqliteSaver checks this but aiosqlite doesn't have it
        setattr(conn, "is_alive", lambda: True)
        
        checkpointer = AsyncSqliteSaver(conn)
        app.state.agent = workflow.compile(checkpointer=checkpointer)
        yield
    print("Server Shutdown.")

app = FastAPI(title="VIBE Agent API", lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---

class ChatRequest(BaseModel):
    message: str
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

@app.get("/history/{thread_id}")
async def get_history(thread_id: str, request: Request):
    """Retrieves conversation history from the persistent graph state."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        # Use compiled agent from app state
        if not hasattr(request.app.state, "agent"):
             raise HTTPException(status_code=500, detail="Agent not initialized")
             
        state = await request.app.state.agent.aget_state(config)
        
        if not state.values:
            return {"messages": []}
            
        messages = state.values.get("messages", [])
        
        # Serialize messages
        history = []
        for msg in messages:
            role = "user"
            if isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                continue # Skip system messages
                
            history.append({
                "role": role,
                "content": msg.content
            })
            
        return {"messages": history}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- Streaming Logic ---

async def event_generator(message: str, thread_id: str, app_state) -> AsyncGenerator[str, None]:
    """
    Streams events from the LangGraph agent to the client using Server-Sent Events (NDJSON).
    
    Features:
    - **Trace Events**: Signals start/end of nodes (for UI spinners).
    - **Plan Updates**: Streams changes to the agent's plan/steps.
    - **Token Streaming**: Streams LLM tokens in real-time (hiding internal thought processes).
    - **Client Actions**: Triggers UI events like 'show_download' or 'outline_update'.
    
    Args:
        message: The user's input message.
        thread_id: The session/thread ID for persistence.
        app_state: The FastAPI app state containing the compiled agent.
    """
    try:
        if not hasattr(app_state, "agent"):
             yield json.dumps({"type": "error", "error": "Agent not initialized"}) + "\n"
             return

        agent_app = app_state.agent
        
        # When using persistent checkpointer, we just send the NEW message.
        # LangGraph appends it to existing state automatically.
        inputs = {
             "messages": [HumanMessage(content=message)],
             "session_id": thread_id
        }
        
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
        
        # Track the active node to decide which tokens to stream
        active_node = None
        # Buffer for outline detection
        line_buffer = ""

        # Use astream_events v2 for granular control
        async for event in agent_app.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]
            name = event["name"]
            data = event.get("data", {})
            
            # 1. Tracing (Node Start)
            if kind == "on_chain_start" and event.get("parent_ids"):
                 # Filter for LangGraph nodes (usually have parent_ids and are in the 'nodes' namespace in metadata)
                 metadata = event.get("metadata", {})
                 is_node = metadata.get("langgraph_node") is not None
                 if is_node:
                    active_node = name
                    yield json.dumps({
                        "type": "trace",
                        "node": name,
                        "step": "start",
                        "icon": metadata.get("ui_icon", "🔄")
                    }) + "\n"

            # 2. Planning Update & Metadata
            if kind == "on_chain_end":
                output = data.get("output", {})
                if not output: continue 

                # Capture Metadata (Step, Plan, Status) immediately
                if event.get("parent_ids"):
                    if isinstance(output, dict) and "plan" in output:
                         yield json.dumps({
                            "type": "plan",
                            "content": output["plan"]
                        }) + "\n"

                    if isinstance(output, dict) and "is_plan_done" in output:
                        yield json.dumps({
                            "type": "plan_status",
                            "is_done": output["is_plan_done"]
                        }) + "\n"

                    if isinstance(output, dict) and "active_step_index" in output:
                        yield json.dumps({
                            "type": "active_step",
                            "index": output["active_step_index"],
                            "description": output.get("active_step_description", ""),
                            "icon": metadata.get("ui_icon", "🎯")
                        }) + "\n"

                # Capture Final Actions from the root graph (no parent)
                if not event.get("parent_ids"): 
                    final_actions = output.get("client_actions", [])
                    yield json.dumps({
                        "type": "final",
                        "client_actions": final_actions,
                        "thread_id": thread_id
                    }) + "\n"

                # Capture Worker Results
                if isinstance(output, dict) and name not in ["planner", "supervisor"]:
                    msgs = output.get("messages", [])
                    if msgs and len(msgs) > 0 and hasattr(msgs[-1], "content"):
                        yield json.dumps({
                            "type": "node_result",
                            "node": name,
                            "content": msgs[-1].content
                        }) + "\n"
            
            # 3. Tool Events (Observability Layer)
            if kind == "on_tool_end":
                # Check what tool finished
                tool_output = data.get("output", "")
                
                # B) Download Button: save_file, save_as_pdf
                if name in ["save_file", "save_as_pdf"]:
                    filename = None
                    
                    # 1. Try to get filename from input args (Most reliable)
                    tool_input = data.get("input", {})
                    # Input might be a dict or a string (json) depending on how it's passed
                    if isinstance(tool_input, dict):
                         filename = tool_input.get("filename")
                    
                    # 2. Fallback: Parse output string
                    if not filename:
                        import re
                        match = re.search(r"to (.*)$", str(tool_output))
                        if match:
                            filepath = match.group(1).strip()
                            filename = filepath.split("/")[-1].split("\\")[-1] # Robust split
                            
                    if filename:
                        yield json.dumps({
                            "type": "server_action",
                            "action": "show_download",
                            "filename": filename
                        }) + "\n"

            # 4. Token Streaming & Outline Detection
            if kind == "on_chat_model_stream":
                if active_node not in ["supervisor", "planner"]:
                    chunk = data.get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        
                        # C) Live Outline Detection (Only from Formatter)
                        if active_node == "Formatter":
                            line_buffer += content
                            if "\n" in line_buffer:
                                lines = line_buffer.split("\n")
                                # Process all complete lines
                                for line in lines[:-1]:
                                    stripped = line.strip()
                                    if stripped.startswith("#"):
                                        # It's a header!
                                        yield json.dumps({
                                            "type": "server_action",
                                            "action": "outline_update",
                                            "data": {"header": stripped}
                                        }) + "\n"
                                # Keep the last partial line
                                line_buffer = lines[-1]

                        # D) STREAMING FILTER: Hide internal tokens from Supervisor/Planner (JSON noise)
                        # We use explicit tags set in the graph definition
                        tags = event.get("tags", [])
                        if "hidden" not in tags and active_node not in ["supervisor", "planner", "supervisor_node", "planner_node"]:
                             yield json.dumps({
                                "type": "token",
                                "content": content
                            }) + "\n"

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"ERROR in event_generator: {error_msg}")
        yield json.dumps({
            "type": "error",
            "content": str(e),
            "trace": error_msg
        }) + "\n"

# --- Endpoints ---

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, req: Request):
    return StreamingResponse(event_generator(request.message, request.thread_id, req.app.state), media_type="application/x-ndjson")

    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads and indexes a file (PDF, TXT, MD) into the Vector Store.
    
    Process:
    1. Detects file type.
    2. Extracts text (uses pypdf for PDFs).
    3. Calls 'index_content' tool to embed and store chunks in ChromaDB.
    """
    filename = file.filename
    content = ""
    
    try:
        # Read file content
        file_bytes = await file.read()
        
        if filename.lower().endswith(".pdf"):
            # Parse PDF
            pdf_stream = io.BytesIO(file_bytes)
            reader = pypdf.PdfReader(pdf_stream)
            text_list = []
            for page in reader.pages:
                text_list.append(page.extract_text())
            content = "\n".join(text_list)
            
        elif filename.lower().endswith((".txt", ".md", ".markdown")):
            # Parse Text
            content = file_bytes.decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, TXT, or MD.")
            
        if not content.strip():
             raise HTTPException(status_code=400, detail="Empty file content.")
             
        # Index the content
        # We use the tool directly. Note: tools returns a string output usually.
        # We construct a synthetic source.
        source = f"Uploaded File: {filename}"
        
        # Call the tool synchronously (since it's a wrapper around sync code or async? 
        # index_content is a LangChain tool. We can invoke it.)
        result = index_content.invoke({"content": content, "source": source})
        
        return {"status": "success", "filename": filename, "result": str(result), "size": len(content)}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
