import requests

def main():
    url = "http://localhost:8000/v1/audios/generate"
    data = {
        "request_id": "123",
        "input_text": "Hello, how are you?",
        "voice_type": "male"
    }
    response = requests.post(url, json=data)
    print(response.json())

if __name__ == "__main__":
    main()
