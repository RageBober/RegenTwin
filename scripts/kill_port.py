"""Kill any process listening on a given port (Windows)."""
import subprocess
import sys

port = sys.argv[1] if len(sys.argv) > 1 else "8000"

try:
    out = subprocess.check_output("netstat -ano", shell=True).decode(errors="ignore")
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line:
            pid = line.strip().split()[-1]
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
except Exception:
    pass
