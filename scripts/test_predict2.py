import subprocess, time, json, sys, os, requests, signal

# Start FastAPI server as a subprocess (non-blocking)
uvicorn_cmd = [sys.executable, '-m', 'uvicorn', 'api.main:app', '--host', '127.0.0.1', '--port', '8000']
proc = subprocess.Popen(uvicorn_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Wait for server to be ready (simple wait, could be improved)
time.sleep(5)

# Load payload JSON
payload_path = os.path.join(os.path.dirname(__file__), '..', 'payload.json')
with open(payload_path, 'r') as f:
    payload = json.load(f)

# Send POST request
try:
    resp = requests.post('http://127.0.0.1:8000/api/predict', json=payload, timeout=10)
    print('Status code:', resp.status_code)
    print('Response:', resp.json())
except Exception as e:
    print('Request failed:', e)

# Terminate server
proc.terminate()
try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()

