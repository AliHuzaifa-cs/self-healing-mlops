"""
Failure type 5: Container failure.
Uses Docker CLI (via subprocess) to stop the API container,
simulating a crash. Docker's `restart: unless-stopped` policy
should bring it back automatically -- OR our self-healing system
will detect and restart it (Stage 8).
"""

import subprocess

CONTAINER_NAME = "luck-prediction-api"


def kill_container():
    print(f"Stopping container: {CONTAINER_NAME}...")
    result = subprocess.run(["docker", "kill", CONTAINER_NAME], capture_output=True, text=True)
    if result.returncode == 0:
        print("Container killed successfully.")
    else:
        print(f"Error: {result.stderr}")


if __name__ == "__main__":
    kill_container()