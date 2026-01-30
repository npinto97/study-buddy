
import requests
import json

try:
    response = requests.get("http://localhost:4040/api/tunnels")
    data = response.json()
    tunnels = data.get("tunnels", [])
    if tunnels:
        url = tunnels[0]['public_url']
        with open("ngrok_url.txt", "w") as f:
            f.write(url)
    else:
        with open("ngrok_url.txt", "w") as f:
            f.write("No active tunnels found.")
except Exception as e:
    with open("ngrok_url.txt", "w") as f:
        f.write(f"Error: {e}")
