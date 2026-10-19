"""Minimal client for the local jev-qwen API."""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


with urllib.request.urlopen(BASE + "/health") as r:
    print("health:", json.load(r))
payload = json.load(open("examples/request.json"))
print(json.dumps(post("/v1/decide", payload), indent=2))
