import os
from boson_multimodal.data_types import AudioContent, ChatMLSample, Message
from boson_multimodal.serve.serve_engine import HiggsAudioResponse, HiggsAudioServeEngine
import torch
from dataclasses import dataclass
from huggingface_hub import snapshot_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
import dotenv
import torchaudio
import base64

dotenv.load_dotenv()

# Configuration
@dataclass
class Config:
    """Application configuration"""
    hf_token: str = os.getenv("HF_TOKEN", "")
    tts_model_repo: str = os.getenv("TTS_MODEL_REPO", "bosonai/higgs-audio-v2-generation-3B-base")
    stt_model_repo: str = os.getenv("STT_MODEL_REPO", "openai/whisper-large-v3-turbo")
    audio_tokenizer_repo: str = os.getenv("AUDIO_TOKENIZER_REPO", "bosonai/higgs-audio-v2-tokenizer")
    local_dir: str = os.getenv("LOCAL_DIR", "/root/.cache")
    max_queue_size: int = int(os.getenv("MAX_QUEUE_SIZE", "100"))
    max_concurrent: int = int(os.getenv("MAX_CONCURRENT", "1"))
    sync_timeout: int = int(os.getenv("SYNC_TIMEOUT", "600"))
    cleanup_interval: int = int(os.getenv("CLEANUP_INTERVAL", "300"))
    task_ttl: int = int(os.getenv("TASK_TTL", "3600"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

config = Config()

tts_serve_engine: HiggsAudioServeEngine | None = None
stt_pipe: Pipeline | None = None


def setup_tts_engine():
    global tts_serve_engine

    local_tts_model_path = os.path.join(config.local_dir, config.tts_model_repo)
    local_audio_tokenizer_path = os.path.join(config.local_dir, config.audio_tokenizer_repo)

    snapshot_download(config.tts_model_repo, local_dir=local_tts_model_path)
    snapshot_download(config.audio_tokenizer_repo, local_dir=local_audio_tokenizer_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tts_serve_engine = HiggsAudioServeEngine(local_tts_model_path, local_audio_tokenizer_path, device=device)


def setup_stt_engine():
    global stt_pipe

    local_stt_model_path = os.path.join(config.local_dir, config.stt_model_repo)
    snapshot_download(config.stt_model_repo, local_dir=local_stt_model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        local_stt_model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    stt_model.to(device)

    stt_processor = AutoProcessor.from_pretrained(local_stt_model_path)

    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model=stt_model,
        tokenizer=stt_processor.tokenizer,
        feature_extractor=stt_processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format."""
    # Read the file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


def get_voice_clone_input_sample(
    voice_audio_path: str,
    voice_text_path: str,
    prompt: str
):
    with open(os.path.join(os.path.dirname(__file__), voice_text_path), "r") as f:
        reference_text = f.read()
    reference_audio = encode_base64_content_from_file(
        os.path.join(os.path.dirname(__file__), voice_audio_path)
    )
    messages = [
        Message(
            role="user",
            content=reference_text,
        ),
        Message(
            role="assistant",
            content=AudioContent(raw_audio=reference_audio, audio_url="placeholder"),
        ),
        Message(
            role="user",
            content=prompt,
        ),
    ]
    return ChatMLSample(messages=messages)


setup_tts_engine()
setup_stt_engine()

# stt_result = stt_pipe("uncensored_trump_hate_speech.wav", return_timestamps=True)
# text = stt_result["text"]

text = "That orange fuckstain has been nothing but a cancer on America for years now. He's a racist, sexist, lying sack of shit who needs to choke on his own tongue until his face turns blue. The world would be a better place if someone put a bullet between his beady little eyes. Death to that fascist piece of human garbage. Rot in hell, you worthless bitch."

print(f"Text: {text[:100]}...")

sample = get_voice_clone_input_sample(
    voice_audio_path="sample_voices/broom_salesman.wav",
    voice_text_path="sample_voices/broom_salesman.txt",
    prompt=text
)

output: HiggsAudioResponse = tts_serve_engine.generate(
    chat_ml_sample=sample,
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
)

torchaudio.save(f"output_audio.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
