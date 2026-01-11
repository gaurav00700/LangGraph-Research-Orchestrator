import requests
import json
import uuid
import time

API_URL = "http://127.0.0.1:8000/chat"

def test_loop_repro():
    thread_id = f"repro-{uuid.uuid4()}"
    
    # 1. First request to set up a plan
    print("--- First Request: Research AI ---")
    payload1 = {"message": "Research AI in healthcare", "thread_id": thread_id}
    res1 = requests.post(API_URL, json=payload1, stream=True)
    for line in res1.iter_lines():
        if line:
            event = json.loads(line)
            if event["type"] == "plan":
                print(f"Plan: {event['content'][:50]}...")
            if event["type"] == "final":
                print("First request finished.")
                break

    time.sleep(2)
    
    # 2. Second request: Simple 'Hi'
    print("\n--- Second Request: Hi ---")
    payload2 = {"message": "Hi", "thread_id": thread_id}
    res2 = requests.post(API_URL, json=payload2, stream=True)
    
    token_count = 0
    start_time = time.time()
    for line in res2.iter_lines():
        if line:
            event = json.loads(line)
            if event["type"] == "token":
                token_count += 1
                if token_count % 100 == 0:
                    print(f"Received {token_count} tokens...")
            if event["type"] == "trace":
                 print(f"Node: {event['node']}")
            if event["type"] == "final":
                print("Second request finished.")
                break
        
        if time.time() - start_time > 60:
            print("Timed out (likely looping)!")
            break

    print(f"Total tokens for 'Hi': {token_count}")

if __name__ == "__main__":
    test_loop_repro()
