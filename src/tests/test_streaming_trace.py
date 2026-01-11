import requests
import json
import uuid

# Configuration
API_URL = "http://127.0.0.1:8000/chat"

def test_streaming_trace_and_results():
    thread_id = str(uuid.uuid4())
    goal = "Search for 1 paper on AI and save it"
    
    print(f"\n1. Testing Streaming for goal: {goal}")
    payload = {"message": goal, "thread_id": thread_id}
    
    # Track which event types we see
    seen_types = set()
    
    try:
        response = requests.post(API_URL, json=payload, stream=True, timeout=90)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                event = json.loads(line.decode("utf-8"))
                e_type = event.get("type")
                seen_types.add(e_type)
                
                if e_type == "trace":
                    print(f"   -> [TRACE] Node: {event.get('node')}")
                elif e_type == "plan":
                    print(f"   -> [PLAN] Received plan steps.")
                elif e_type == "node_result":
                    print(f"   -> [RESULT] Node {event.get('node')} finished.")
                elif e_type == "token":
                    # Print first few tokens only
                    if "token" not in [t for t in seen_types if t != "token"]:
                         print(f"   -> [TOKEN] Streaming started...")
                elif e_type == "final":
                    print(f"   -> [FINAL] Actions: {event.get('client_actions')}")
                elif e_type == "error":
                    print(f"   -> [ERROR] {event.get('content')}")

    except Exception as e:
        print(f"   !! Failed: {str(e)}")
        raise e

    # Assertions
    assert "trace" in seen_types, "Should have seen trace events"
    assert "plan" in seen_types, "Should have seen a plan"
    assert "node_result" in seen_types, "Should have seen node results (Researcher/Formatter)"
    assert "final" in seen_types, "Should have reached final event"
    
    print("\nStreaming verification successful!")

if __name__ == "__main__":
    test_streaming_trace_and_results()
