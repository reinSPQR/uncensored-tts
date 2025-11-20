import requests

def check_health():
    url = "http://localhost:8000/health"
    response = requests.get(url)
    print(response.json())

def check_status():
    url = "http://localhost:8000/status"
    response = requests.get(url)
    print(response.json())

def shutdown():
    url = "http://localhost:8000/shutdown"
    response = requests.get(url, headers={"X-API-Key": "super-secret"})
    print(response.json())

if __name__ == "__main__":
    check_status()
