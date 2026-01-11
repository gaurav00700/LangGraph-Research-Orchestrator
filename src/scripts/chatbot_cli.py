"""
Agent Command Line Interface.

This script allows direct interaction with the LangGraph multi-agent system from 
the terminal. It supports real-time event streaming, node tracing, and 
session persistence via a local SQLite checkpointer.
"""

import asyncio
import uuid
import sys
import os
import json
import argparse
import aiosqlite

# Add root to pass to find 'agent'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent.graph import workflow, DB_PATH
from agent.logging_utils import get_log_path

async def run_cli():
    parser = argparse.ArgumentParser(description="VIBE Agent CLI")
    parser.add_argument("--session-id", type=str, help="Resume an existing session by ID")
    args = parser.parse_args()

    print("🚀 Multi Agent CLI (Type 'quit' to exit)")
    
    if args.session_id:
        thread_id = args.session_id
        print(f"Resuming Session: {thread_id}")
    else:
        thread_id = str(uuid.uuid4())
        print(f"New Session ID: {thread_id}")
        
    print(f"Log File: {get_log_path(thread_id)}\n")

    async with aiosqlite.connect(DB_PATH) as conn:
        setattr(conn, "is_alive", lambda: True)
        checkpointer = AsyncSqliteSaver(conn)
        agent_app = workflow.compile(checkpointer=checkpointer)

        # Load history if resuming
        existing_messages = []
        if args.session_id:
            config = {"configurable": {"thread_id": thread_id}}
            state = await agent_app.aget_state(config)
            if state.values:
                existing_messages = state.values.get("messages", [])
                print("--- Previous Conversation ---")
                for m in existing_messages:
                    role = "You" if isinstance(m, HumanMessage) else "Agent"
                    # Filter out internal supervisor routing notes from display if they exist
                    if not (isinstance(m, AIMessage) and any(x in str(m.content) for x in ["Supervisor Routing", "ROUTING"])):
                        print(f"{role}: {m.content}")
                print("-----------------------------\n")

        while True:
            try:
                # print("\n----------------------")
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit"]:
                    break
                
                # Start with history + new message? 
                # With persistent checkpointer, we just send valid inputs (which is list of messages)
                # React agent state handles appending.
                
                inputs = {
                    "messages": [HumanMessage(content=user_input)],
                    "session_id": thread_id
                }
                
                # Note: We don't need to manually pass 'existing_messages' + new message
                # because the graph state already has it. passing just the new one is LangGraph standard for persistent graphs.
                # However, for the display logic below we aren't using 'inputs' for display anyway.
                config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
                
                
                # Keep track of printed items to verify duplicates
                printed_results = set()
                last_step_index = -1
                
                new_ai_content = ""
                
                # Main Output Loop: Consume the LangGraph event stream
                # We use astream_events(version='v2') to get granular updates on node starts,
                # tool calls, and model outputs.
                async for event in agent_app.astream_events(inputs, config=config, version="v2"):
                    kind = event["event"]
                    name = event["name"]
                    data = event.get("data", {})
                    
                    # Filter out noisy internal names
                    if name in ["RunnableSequence", "RunnableLambda", "Unnamed", "agent", "call_model", "should_continue", "LangGraph", "tools", "Prompt"]:
                        continue
    
                    # 1. Tracing (Node Start)
                    if kind == "on_chain_start":
                         # Filter out the wrapper node that has the same name as the graph
                         if name == "LangGraph": continue
                         
                         metadata = event.get("metadata", {})
                         is_node = metadata.get("langgraph_node") is not None
                         if is_node:
                            icon = metadata.get("ui_icon", "🔄")
                             # Only verify if it's one of our main workers
                            if name in ["planner", "supervisor", "Researcher", "AnalystAgent", "Formatter", "ChatAgent", "ContextQA"]:
                                 print(f"{icon} {name} working...")
    
                    # Tool Updates
                    if kind == "on_tool_start":
                         print(f"  🛠️  Calling tool: {name}...")
    
                    # 2. Planning Update
                    if kind == "on_chain_end":
                        output = data.get("output", {})
                        if not output: continue
    
                        if name == "planner" and isinstance(output, dict) and "plan" in output:
                            # Pretty print the initial plan
                            print(f"\n📝 Initial Plan:\n{output['plan']}\n")
                            continue
    
                        if isinstance(output, dict) and "active_step_index" in output:
                            desc = output.get("active_step_description", "")
                            idx = output["active_step_index"]
                            # Only show step update if it changed or has a meaningful description
                            if idx != last_step_index and desc:
                                print(f"\n🎯 {desc}\n")
                                last_step_index = idx
                        
                        # Worker Results
                        if isinstance(output, dict):
                            msgs = output.get("messages", [])
                            if msgs and len(msgs) > 0 and hasattr(msgs[-1], "content"):
                                 content = msgs[-1].content
                                 if content and content.strip():
                                     # De-duplicate: Don't print the exact same content string twice for the same node
                                     # (Simple de-duplication)
                                     sig = f"{name}:{content[:20]}"
                                     if sig not in printed_results:
                                         print(f"\n✅ {name}:\n{content}\n")
                                         printed_results.add(sig)
    
                    # 3. Token Streaming
                    if kind == "on_chat_model_stream":
                         # Simple token streaming
                         chunk = data.get("chunk")
                         if chunk and hasattr(chunk, "content"):
                            sys.stdout.write(chunk.content)
                            sys.stdout.flush()
                
                print("----------------------")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_cli())
