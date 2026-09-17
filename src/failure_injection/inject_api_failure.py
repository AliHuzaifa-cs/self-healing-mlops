"""
Failure type 4: API failure simulation.
Sends malformed/invalid requests to the running API to trigger errors,
and can simulate load to observe latency behavior.
"""

import requests
import time

API_URL = "http://127.0.0.1:8000"


def send_invalid_requests(count=10):
    print(f"Sending {count} malformed requests to /predict...")
    errors = 0
    for i in range(count):
        # Missing required fields -> should trigger FastAPI validation error (422)
        resp = requests.post(f"{API_URL}/predict", json={"return_lag1": 0.01})
        if resp.status_code >= 400:
            errors += 1
    print(f"{errors}/{count} requests returned error status codes (expected).")


def simulate_load(count=200, delay=0.01):
    print(f"Simulating {count} rapid requests to /health...")
    start = time.time()
    for _ in range(count):
        requests.get(f"{API_URL}/health")
        time.sleep(delay)
    duration = time.time() - start
    print(f"Completed {count} requests in {duration:.2f}s ({count/duration:.1f} req/s)")


if __name__ == "__main__":
    send_invalid_requests()
    simulate_load()