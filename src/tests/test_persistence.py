import requests
import json
import uuid
import time
import pytest

# Configuration
API_URL = "http://127.0.0.1:8000/chat"

def get_events(message, thread_id):
    payload = {"message": message, "thread_id": thread_id}
    response = requests.post(API_URL, json=payload, stream=True, timeout=90)
    response.raise_for_status()
    events = []
    for line in response.iter_lines():
        if line:
            events.append(json.loads(line.decode("utf-8")))
    return events

def test_plan_persistence():
    thread_id = str(uuid.uuid4())
    
    # 1. Start with a goal
    goal = "Hi"
    print(f"\n1. Sending Goal: {goal}")
    events1 = get_events(goal, thread_id)
    
    # Actually "Hi" might route to ChatAgent and not generate a plan.
    # Let's use a simple goal that DOES generate a plan.
    goal = "Search for 'test'"
    print(f"\n1. Sending Goal: {goal}")
    events1 = get_events(goal, thread_id)
    
    seen_plan = False
    for event in events1:
        if event["type"] == "plan":
            seen_plan = True
            print(f"   -> Plan received: {event['content'][:50]}...")
            break
    
    assert seen_plan, "Plan should be generated for the first goal"
    
    # 2. Send a follow-up that SHOULD NOT trigger a new plan
    followup = "Tell me more about it"
    print(f"2. Sending Follow-up: {followup}")
    events2 = get_events(followup, thread_id)
    
    triggered_new_plan = False
    seen_trace_supervisor = False
    for event in events2:
        if event["type"] == "plan":
            triggered_new_plan = True
        if event["type"] == "trace" and event["node"] == "supervisor":
             seen_trace_supervisor = True
    
    assert not triggered_new_plan, "Follow-up should NOT trigger a new plan (it should persist)"
    assert seen_trace_supervisor, "Should have routed to supervisor to continue"
    print("   -> Success: Plan persisted and routed correctly.")

if __name__ == "__main__":
    # We expect the server to be running
    try:
        test_plan_persistence()
    except Exception as e:
        print(f"Test failed: {e}")
        exit(1)
