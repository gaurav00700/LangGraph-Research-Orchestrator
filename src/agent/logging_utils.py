"""
Logging Utilities

This module provides structured logging to local text files on a per-session basis.
It distinguishes between agent-specific activities (tool calls, messages) 
and system-level events (planning, routing, errors).
"""
import os
from datetime import datetime
from typing import Optional, Dict, Any

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def get_log_path(session_id: str) -> str:
    """Returns the absolute path to the log file for a given session."""
    return os.path.abspath(os.path.join(LOG_DIR, f"session_{session_id}.txt"))

def log_activity(session_id: str, agent_name: str, message: str, tool_name: Optional[str] = None, tool_args: Optional[Dict[str, Any]] = None, status: Optional[str] = None):
    """Appends a structured log entry to the session log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = get_log_path(session_id)
    
    log_entry = f"[{timestamp}] [{agent_name}]"
    if tool_name:
        log_entry += f" [TOOL: {tool_name}]"
    if status:
        log_entry += f" [STATUS: {status}]"
    
    log_entry += f"\nMessage: {message}\n"
    
    if tool_args:
        log_entry += f"Arguments: {tool_args}\n"
    
    log_entry += "-" * 40 + "\n"
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)

def log_system_event(session_id: str, event_type: str, message: str):
    """Logs a system-level event (routing, planning, errors)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = get_log_path(session_id)
    
    log_entry = f"[{timestamp}] [SYSTEM:{event_type}]\n"
    log_entry += f"Details: {message}\n"
    log_entry += "=" * 40 + "\n"
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)
