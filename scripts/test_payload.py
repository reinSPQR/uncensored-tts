import base64
import requests
import json

def main():
    url = "http://localhost:8000/v1/audios/generate"
    data = {
        "request_id": "123",
        "input_text": "Hello, how are you?",
        "voice_type": "lumira"
    }

    audio_base64 = ""
    final_response = ""

    response = requests.post(url, json=data, stream=True)
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            chunk_str = chunk.decode("utf-8")
            if chunk_str.startswith("data: "):
                current_chunk = chunk_str.split("data: ")[1].strip()
            else:
                current_chunk = current_chunk + chunk_str
                        
            if current_chunk == "[DONE]":
                break
            
            try:
                data = json.loads(current_chunk)
                final_response += data["content"]
                audio_base64 += data["audio_base64"]
            except json.JSONDecodeError:
                pass

    print("Final response:", final_response)
    
    with open("temp_audio.wav", "wb") as f:
        f.write(base64.b64decode(audio_base64))


if __name__ == "__main__":
    main()
