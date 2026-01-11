"""
This module uses LangGraph to define a multi-agent orchestration workflow.
It includes nodes for Planning, Research, Analysis, Formatting, and Chat, 
coordinated by a dynamic Supervisor. State is persisted via SQLite.
"""
from typing import TypedDict, Literal, List, Annotated, Dict
import os, sys
import json
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Ensure the root directory is in sys.path for direct script execution
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)
if os.path.join(root_dir, "src") not in sys.path:
    sys.path.append(os.path.join(root_dir, "src"))

# --- Setup Persistent Checkpoint Path ---
DATA_DIR = os.path.join(root_dir, "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

DB_PATH = os.path.join(DATA_DIR, "checkpoints.db")
# memory = AsyncSqliteSaver.from_conn_string(DB_PATH) # Moved to server/client lifecycle

from agent.tools import (
    save_file, get_arxiv_details, save_as_pdf, read_url,
    index_content, search_vector_store, search_hn, search_arxiv
)
from agent.prompts import (
    SYSTEM_CONTEXT, SUPERVISOR_PROMPT, PLANNER_PROMPT, RESEARCHER_PROMPT,
    ANALYST_PROMPT, FORMATTER_PROMPT, VECTOR_MANAGER_PROMPT, CHAT_PROMPT
)
from agent.logging_utils import log_activity, log_system_event
from langsmith import traceable

# Load env
load_dotenv()

# --- Observability Check ---
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("WARNING: LANGCHAIN_TRACING_V2 is true but LANGCHAIN_API_KEY is missing. Tracing may fail.")
    else:
        print(f"✅ LangSmith Tracing Enabled. Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
else:
    print("ℹ️ LangSmith Tracing is DISABLED. Set LANGCHAIN_TRACING_V2=true to enable.")

# --- Configuration ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
# llm = ChatOllama(model="llama3.1:8b", temperature=0, streaming=True)

# --- State Definition ---
def add_actions(left: List[Dict], right: List[Dict]) -> List[Dict]:
    """Reducer for client_actions list."""
    if not left: left = []
    if not right: right = []
    return left + right

class AgentState(TypedDict):
    """
    Core state for the LangGraph agent workflow.
    
    Attributes:
        messages: The conversation history (list of BaseMessage).
        plan: The current high-level plan string.
        active_step_index: Index of the current step in the plan.
        active_step_description: Goal of the current step.
        current_step_retries: Counter for retries on the same step to prevent infinite loops.
        is_plan_done: Boolean flag indicating if the plan is fully executed.
        client_actions: List of side-effect actions (like file downloads) for the client.
        next: The name of the next worker node to route to.
        session_id: The session ID for logging purposes.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    plan: str
    active_step_index: int
    active_step_description: str # The current goal description for UI
    current_step_retries: int # Track retries for the active step to prevent loops
    is_plan_done: bool
    client_actions: Annotated[List[Dict], add_actions] # List of actions like {"type": "show_download", "path": "..."}
    next: str
    session_id: str # Added for persistent logging

# --- Structured Output for Planner ---
class PlanResponse(BaseModel):
    plan_steps: List[str] = Field(..., description="Ordered list of steps to achieve the goal.")
    initial_worker: Literal["Researcher", "VectorStoreAgent", "ChatAgent", "Formatter"] = Field(
        ..., description="The first worker to start the task."
    )
    is_continuation: bool = Field(
        False, description="True if this is a follow-up or refinement of the previous conversation. False if it's a completely new, unrelated topic."
    )

# --- Structured Output for Supervisor ---
class RouterResponse(BaseModel):
    next_worker: Literal["Researcher", "AnalystAgent", "Formatter", "ChatAgent", "VectorStoreAgent", "planner", "FINISH"] = Field(
        ..., description="The next worker to handle the user's request."
    )
    active_step_index: int = Field(..., description="The index of the current plan step being executed.")
    active_step_description: str = Field(..., description="The text description of the current step from the plan.")
    reasoning: str = Field(..., description="Concise reasoning (max 2 sentences). MUST reference specific evidence in history.")

# --- Prompts ---
# Common instruction to prevent agents from hallucinating limitations or providing manual workarounds


# --- Sub-Agents ---
researcher_agent = create_react_agent(llm, tools=[search_hn, search_arxiv, get_arxiv_details, read_url])
analyst_agent = create_react_agent(llm, tools=[]) # Consolidated reasoning
formatter_agent = create_react_agent(llm, tools=[save_file, save_as_pdf])
vector_manager_agent = create_react_agent(llm, tools=[index_content, search_vector_store])

# --- Nodes ---
def strip_old_history(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Filters messages to only include those from the current 'Task Isolation' block.
    
    The Planner inserts a SystemMessage marker ('--- NEW TASK START') whenever 
    a new goal is defined. This allows workers to focus on the current task 
    without being distracted by previous, unrelated conversations in the thread.
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], SystemMessage) and "--- NEW TASK START" in str(messages[i].content):
            return messages[i:]
    return messages


def parse_and_log_tools(session_id: str, agent_name: str, messages: List[BaseMessage]):
    """Parses messages returned by an agent to log tool calls and results."""

    for i, m in enumerate(messages):
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                # Find corresponding tool output
                output = "No output found."
                status = "Unknown"
                for next_m in messages[i+1:]:
                    if isinstance(next_m, ToolMessage) and next_m.tool_call_id == tc['id']:
                        output = str(next_m.content)
                        status = "SUCCESS" if "error" not in output.lower() else "FAILED"
                        break
                
                log_activity(
                    session_id=session_id,
                    agent_name=agent_name,
                    message=f"Output: {output[:500]}...",
                    tool_name=tc['name'],
                    tool_args=tc['args'],
                    status=status
                )
        elif isinstance(m, AIMessage) and not m.tool_calls:
             log_activity(
                session_id=session_id,
                agent_name=agent_name,
                message=m.content
            )

@traceable(run_type="chain", name="planner_node")
def planner_node(state: AgentState):
    """Generates a plan if one doesn't exist, evaluates if a new plan is needed, or routes to chat."""
    messages = state['messages']
    last_msg = messages[-1].content
    current_plan = state.get("plan", "")
    is_plan_done = state.get("is_plan_done", False)

    planner_prompt = PLANNER_PROMPT.format(
        current_plan=current_plan if current_plan else 'None',
        is_plan_done=is_plan_done
    )
    
    structured_planner = llm.with_structured_output(PlanResponse)
    response = structured_planner.invoke(
        [SystemMessage(content=planner_prompt), HumanMessage(content=last_msg)],
        config={"tags": ["hidden"]}
    )
    
    sid = state.get("session_id", "unknown")
    log_system_event(sid, "PLANNING", f"Planner called with goal: {last_msg}")
    
    # Save the user's turn
    # save_turn(sid, "human", last_msg) # Handled by LangGraph state

    # If the LLM didn't return steps, determine if we should keep existing plan or route to Chat
    if not response.plan_steps:
        if current_plan and not is_plan_done:
            return {"next": "supervisor"}
        return {
            "next": "ChatAgent", 
            "is_plan_done": True,
            "active_step_index": -1, # Reset index to avoid UI artifacts
            "active_step_description": ""
        }

    # If steps were returned, it's a new plan
    plan_str = "\n".join([f"- {s}" for s in response.plan_steps])
    
    # Context Preservation: Only start a "New Task Block" if it's NOT a continuation
    new_messages = []
    if not response.is_continuation:
        marker = SystemMessage(content=f"--- NEW TASK START: {last_msg} ---")
        new_messages.append(marker)
        log_system_event(sid, "PLAN_CREATED", f"New Goal: {last_msg}")
    else:
        log_system_event(sid, "PLAN_CONTINUED", f"Refining goal: {last_msg}")
    
    return {
        "messages": new_messages,
        "plan": plan_str,
        "active_step_index": 0,
        "active_step_description": response.plan_steps[0], # Explicitly lead with Step 1 description
        "current_step_retries": 0,
        "is_plan_done": False,
        "next": response.initial_worker,
        "client_actions": [{"type": "update_plan", "content": plan_str}],
        "session_id": sid # Ensure it's passed if it was in state
    }

@traceable(run_type="chain", name="supervisor_node")
def supervisor_node(state: AgentState):
    """Dynamic State-Based Supervisor."""
    
    # Get state variables
    messages = state['messages']
    plan = state.get('plan', '')
    is_plan_done = state.get("is_plan_done", False)
    active_step_index = state.get("active_step_index", 0)
    
    # 1. Initialization Check: If no plan, or PREVIOUS task is done, route to Planner for new goal
    if not plan or is_plan_done:
         return {"next": "planner"}

    last_ai_msgs = [m.content for m in messages if isinstance(m, AIMessage)]
    last_action = last_ai_msgs[-1] if last_ai_msgs else "None"
    
    system_prompt = SUPERVISOR_PROMPT.format(
        active_step_index=active_step_index,
        active_step_description=state.get('active_step_description', 'Execute Plan'),
        plan=plan,
        last_action=last_action
    )
    
    structured_llm = llm.with_structured_output(RouterResponse)
    # Filter history to keep context clean but DO NOT blindly slice (breaking tool chains)
    relevant_msgs = strip_old_history(messages)
    response = structured_llm.invoke(
        [SystemMessage(content=system_prompt)] + relevant_msgs,
        config={"tags": ["hidden"]}
    )
    
    # --- Logging & Tracing ---
    # We log the supervisor's decision to a file for debugging/audit purposes.
    sid = state.get("session_id", "unknown")
    log_system_event(sid, "ROUTING", f"Supervisor reasoning: {response.reasoning}\nNext Worker: {response.next_worker}")
    
    # --- Safety Overrides & Validation Logic ---
    # The LLM supervisor can sometimes make logical errors (e.g. routing to Formatter 
    # before data is selected). These overrides enforce strict sequential dependencies.
    
    has_content = False
    has_selection = False
    for m in relevant_msgs:
        content_str = str(m.content)
        # Check if we have actual research data in history
        if any(keyword.lower() in content_str.lower() for keyword in ["# Aggregated Reports", "Aggregated Reports", "CONTENT", "Abstract:", "Paper 1:", "Title:"]):
            has_content = True
        # Check if the Analyst has explicitly marked a 'SELECTED:' block for the Formatter
        if "SELECTED:" in content_str.upper() or "SELECTED ARTICLES" in content_str.upper() or "SELECTED TOPIC" in content_str.upper():
            has_selection = True
            
    # Dependency Check A: Formatter requires Analyst selection
    if response.next_worker == "Formatter" and not has_selection:
         print(f"DEBUG: [Safety Override] Formatter called without selection. Routing to AnalystAgent.")
         response.next_worker = "AnalystAgent"
         response.reasoning = "[SAFETY OVERRIDE] Selection is missing. Analyst must select before Formatter can save."
         
    # Dependency Check B: Analyst requires Research content
    if response.next_worker == "AnalystAgent" and not has_content:
        last_output = str(last_action).upper()
        # If we failed to fetch content after multiple tries, escalate to prompt the human
        if "HUMAN_INPUT_REQUIRED" in last_output or "CONTENT UNAVAILABLE" in last_output:
             print(f"DEBUG: [Safety Override] Content retrieval failed. Routing to FINISH for user input.")
             response.next_worker = "FINISH"
             response.reasoning = "Content retrieval failed. Requesting human intervention."
        else:
            # Otherwise, force the Researcher to keep looking
            print(f"DEBUG: [Safety Override] LLM tried to route to {response.next_worker} without content. Forcing Researcher.")
            response.next_worker = "Researcher"
            response.reasoning = "[SAFETY OVERRIDE] Redirecting to Researcher because actual content is missing from task history."


    active_description = response.active_step_description

    print(f"DEBUG: Supervisor State Routing -> {response.next_worker}. Reason: {response.reasoning}")
    
    # --- MECHANICAL STEP LOCK: Prevent Looping ---
    target_index = response.active_step_index
    if response.next_worker != "planner" and response.next_worker != "FINISH":
        # Block backward jumps (except for re-planning)
        if target_index < active_step_index:
            print(f"DEBUG: [Safety Override] Blocked backward jump from {active_step_index} to {target_index}.")
            target_index = active_step_index
        # Block massive forward jumps to keep UI sync
        elif target_index > active_step_index + 1:
            print(f"DEBUG: [Safety Override] Blocked forward jump from {active_step_index} to {target_index}. Forcing sequential step.")
            target_index = active_step_index + 1
            
    # --- PREMATURE FINISH LOCK ---
    # If finishing immediately after planning or with no evidence of work
    if response.next_worker == "FINISH" and len(relevant_msgs) <= 1:
        print(f"DEBUG: [Safety Override] Supervisor tried to FINISH with no history. Routing to Researcher.")
        response.next_worker = "Researcher"
        target_index = 0

    # Retry Logic: If staying on same step/worker, increment. Else reset.
    current_retries = state.get("current_step_retries", 0)
    new_retries = current_retries + 1 if response.next_worker != "FINISH" and target_index == active_step_index else 0
    
    # Force escape if stuck
    if new_retries > 2:
        print(f"DEBUG: Stuck in loop (Retries {new_retries}). Forcing move forward.")
        if response.next_worker == "AnalystAgent":
            response.next_worker = "Researcher" # Let Researcher try to fetch
        else:
            response.next_worker = "AnalystAgent"
        new_retries = 0

    res = {
        "next": response.next_worker,
        "active_step_index": target_index, 
        "active_step_description": active_description,
        "current_step_retries": new_retries
    }
    
    # Log the routing decision for audit
    sid = state.get("session_id", "unknown")
    import json
    log_system_event(sid, "SUPERVISOR_ROUTING", json.dumps(res))
    
    if response.next_worker == "FINISH":
        res["is_plan_done"] = True
        
    return res

@traceable(run_type="chain", name="researcher_node")
def researcher_node(state: AgentState):
    """
    Executes the research phase using search tools.
    
    1. Strips old history to focus on current task.
    2. Invokes the prebuilt React Agent with research tools.
    3. Logs tool usage and results.
    """
    relevant_msgs = strip_old_history(state['messages'])
    messages = [SystemMessage(content=RESEARCHER_PROMPT)] + relevant_msgs
    result = researcher_agent.invoke({"messages": messages})
    # result["messages"] contains the full conversation. Extract NEW messages.
    # We added one SystemMessage at the start, so new ones start after len(messages)
    new_msgs = result["messages"][len(messages):]
    
    parse_and_log_tools(state.get("session_id", "unknown"), "Researcher", new_msgs)
    
    return {"messages": new_msgs}

@traceable(run_type="chain", name="analyst_node")
def analyst_node(state: AgentState):
    """
    Analyzes and summarizes research findings.
    
    1. Uses the Analyst Prompt to evaluate gathered content.
    2. Selects relevant items (must include valid selection for Formatter).
    3. Generates a summary.
    """
    relevant_msgs = strip_old_history(state['messages'])
    messages = [SystemMessage(content=ANALYST_PROMPT)] + relevant_msgs
    result = analyst_agent.invoke({"messages": messages})
    new_msgs = result["messages"][len(messages):]
    
    parse_and_log_tools(state.get("session_id", "unknown"), "AnalystAgent", new_msgs)
    
    return {"messages": new_msgs}
    


@traceable(run_type="chain", name="formatter_node")
def formatter_node(state: AgentState):
    """
    Formats and saves the final output to a file.
    
    1. Invokes Formatter Agent to write the file (md/txt/pdf).
    2. Scans the output for success confirmation.
    3. Emits a 'client_action' (show_download) if a file was saved.
    """
    relevant_msgs = strip_old_history(state['messages'])
    messages = [SystemMessage(content=FORMATTER_PROMPT)] + relevant_msgs
    result = formatter_agent.invoke({"messages": messages})
    new_msgs = result["messages"][len(messages):]
    
    parse_and_log_tools(state.get("session_id", "unknown"), "Formatter", new_msgs)
    
    # Check for client action: if a report was saved
    # Search through new messages for the success phrase
    res_text = ""
    for m in new_msgs:
        if isinstance(m, AIMessage):
            res_text += str(m.content)
            
    actions = []
    if "File saved successfully" in res_text or "PDF saved successfully" in res_text:
        # Extract filename if possible, otherwise default
        import re
        match = re.search(r"to (.*)$", res_text)
        if match:
             filepath = match.group(1).strip()
             filename = os.path.basename(filepath)
        else:
             filename = "report.pdf" if "PDF" in res_text else "report.md"
             
        actions.append({"type": "show_download", "filename": filename})
        actions.append({"type": "confetti"})
        
    return {"messages": new_msgs, "client_actions": actions}


@traceable(run_type="chain", name="chat_node")
def chat_node(state: AgentState):
    """
    Handles general chat interactions (greetings, simple questions).
    
    This node does NOT execute a plan. It simply responds to the user and marks the plan as done.
    """
    response = llm.invoke([SystemMessage(content=CHAT_PROMPT)] + state['messages'])
    
    # save_turn(sid, "ai", response.content) # Handled by LangGraph state

        
    return {"messages": [response], "is_plan_done": True}

@traceable(run_type="chain", name="vector_manager_node")
def vector_manager_node(state: AgentState):
    """
    Manages the Vector Store (ChromaDB) for long-term memory.
    
    Capable of 'Index' (save) and 'Retrieve' (search) operations.
    """
    relevant_msgs = strip_old_history(state['messages'])
    messages = [SystemMessage(content=VECTOR_MANAGER_PROMPT)] + relevant_msgs
    result = vector_manager_agent.invoke({"messages": messages})
    new_msgs = result["messages"][len(messages):]
    
    parse_and_log_tools(state.get("session_id", "unknown"), "VectorStoreAgent", new_msgs)
    
    return {"messages": new_msgs}

# --- Graph Construction ---
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node, metadata={"ui_icon": "🎮"})
workflow.add_node("planner", planner_node, metadata={"ui_icon": "📋"})
workflow.add_node("Researcher", researcher_node, metadata={"ui_icon": "🔍"})
workflow.add_node("AnalystAgent", analyst_node, metadata={"ui_icon": "🧐"})
workflow.add_node("Formatter", formatter_node, metadata={"ui_icon": "📝"})
workflow.add_node("ChatAgent", chat_node, metadata={"ui_icon": "🤖"})
workflow.add_node("VectorStoreAgent", vector_manager_node, metadata={"ui_icon": "📚"})

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "planner", 
    lambda x: x["next"], # Planner now routes directly to initial worker
    {
        "Researcher": "Researcher",
        "ChatAgent": "ChatAgent",
        "VectorStoreAgent": "VectorStoreAgent",
        "Formatter": "Formatter",
        "supervisor": "supervisor"
    }
)

workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "Researcher": "Researcher",
        "AnalystAgent": "AnalystAgent",
        "Formatter": "Formatter",
        "ChatAgent": "ChatAgent",
        "VectorStoreAgent": "VectorStoreAgent",
        "planner": "planner",
        "FINISH": END
    }
)

# Workers go back to supervisor to check plan progress
workflow.add_edge("Researcher", "supervisor")
workflow.add_edge("AnalystAgent", "supervisor")
workflow.add_edge("Formatter", "supervisor")
workflow.add_edge("VectorStoreAgent", "supervisor")
workflow.add_edge("ChatAgent", END)

# app = workflow.compile(checkpointer=memory) # Moved to server/client lifecycle

if __name__ == "__main__":
    # Generate and save the architecture diagram
    from langgraph.checkpoint.memory import MemorySaver
    _app = workflow.compile(checkpointer=MemorySaver())
    
    # 1. Print ASCII for terminal feedback
    print("--- Agent Architecture (ASCII) ---")
    print(_app.get_graph().draw_ascii())
    print("-" * 33)

    # 2. Save PNG to assets folder
    assets_dir = os.path.join(root_dir, "assets")
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        
    try:
        png_path = os.path.join(assets_dir, "architecture.png")
        # LangGraph's draw_mermaid_png returns bytes
        png_bytes = _app.get_graph().draw_mermaid_png()
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        print(f"✅ Architecture diagram saved to: {png_path}")
    except Exception as e:
        print(f"❌ Could not save PNG: {e}")
        print("Tip: Ensure 'pygraphviz' or a valid mermaid rendering environment is available if using local renderers.")
