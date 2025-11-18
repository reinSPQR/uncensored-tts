import base64
import requests
import json

def main():
    url = "http://localhost:8000/v1/audios/generate"
    data = {
        "input_text": "Hello, how are you?",
        "voice_sample_url": "https://storage.googleapis.com/eternal-ai/test-rein/voice-lumira/lumira.mp3",
        "voice_sample_text": "Doctor, have you heard of the Avenue des Champs-Élysées? Yes, I happened to be born there. I really wish I could go back and visit one day... what? Oh, no, I appreciate it, but I'm not lonely. I have you, after all! You're my family."
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
