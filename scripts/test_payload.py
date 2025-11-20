import base64
import requests
import json
import time

def main():
    url = "http://localhost:8000/v1/audios/generate"
    data = {
        # "input_text": "Epic Games and Apple – now there's a saga that's hard to ignore! So here's the latest twist: Apple's new install process has cut user drop-offs by 60%. Now, let’s unpack this. Initially, Apple's installation process was like navigating a labyrinth – full of ominous warnings that scared off 65% of users trying to install Epic's software. But with iOS 18.6, Apple streamlined it, slashing those drop-offs to just 25%. That's the same as what we see on Windows and macOS. Fascinating, right? This tells us something crucial: the barrier wasn't distrust of third-party stores – it was Apple's design! Now, what does this mean for the market? With fewer obstacles, we're likely to see more innovation and competition, which is music to any developer's ears. Yet, despite this progress, Apple still holds tight to certain fees and restrictions. Critics say these are still hurdles, but the security angle can't be ignored – Apple's balancing",
        "input_text": "Hello, how are you?",
        # "voice_sample_url": "https://storage.googleapis.com/eternal-ai/test-rein/voice-lumira/lumira.mp3",
        # "voice_sample_text": "Doctor, have you heard of the Avenue des Champs-Élysées? Yes, I happened to be born there. I really wish I could go back and visit one day... what? Oh, no, I appreciate it, but I'm not lonely. I have you, after all! You're my family."
        "voice_sample_url": "https://storage.googleapis.com/eternal-ai/test-rein/voice-belinda/belinda.wav",
        "voice_sample_text": "Twas the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year."
    }

    audio_base64 = ""
    final_response = ""
    current_chunk = ""

    start_time = time.time()

    iter = 0

    response = requests.post(url, json=data, stream=True)
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            # print(f"Received chunk {iter}: {chunk}")
            
            iter += 1

            chunk_str = chunk.decode("utf-8")
            if chunk_str.startswith("data: "):
                current_chunk = chunk_str.split("data: ")[1].strip()
            else:
                current_chunk = current_chunk + chunk_str
                        
            if current_chunk == "[DONE]":
                break

            try:
                data = json.loads(current_chunk)
                if "content" in data and data["content"]:
                    final_response += data["content"]
                if "audio_base64" in data and data["audio_base64"]:
                    audio_base64 += data["audio_base64"]
                if data["status"]["progress"] < 99.0:
                    print(f"Received progress: {data['status']['progress']}")
            except Exception as e:
                pass

    print(f"TTS generation time: {time.time() - start_time} seconds")

    print("Final response:", final_response)
    with open("temp_audio.wav", "wb") as f:
        f.write(base64.b64decode(audio_base64))


if __name__ == "__main__":
    main()
