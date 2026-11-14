import subprocess, time, json, requests, os, signal, sys

# Path to uvicorn executable within the virtual environment
uvicorn_path = os.path.join(os.path.dirname(__file__), '..', '.venv', 'Scripts', 'uvicorn.exe')

# Start the FastAPI server as a subprocess
proc = subprocess.Popen([uvicorn_path, 'api.main:app', '--host', '127.0.0.1', '--port', '8000'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait a few seconds for the server to start
time.sleep(5)

# Load payload
payload_path = os.path.join(os.path.dirname(__file__), '..', 'payload.json')
with open(payload_path, 'r') as f:
    data = json.load(f)

# Send request
try:
    resp = requests.post('http://127.0.0.1:8000/api/predict', json=data)
    print('Status code:', resp.status_code)
    print('Response JSON:', resp.json())
except Exception as e:
    print('Request failed:', e)

# Terminate the server process
proc.terminate()
proc.wait()

