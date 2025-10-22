WEBHOOK_URL=
WEBHOOK_API_KEY=
BACKEND_UPLOAD_URL=
BACKEND_UPLOAD_ADMIN_KEY=

docker run -p 8000:8000 \
    -v ./checkpoints:/app/checkpoints \
    -v ./hf_cache:/app/hf_cache \
    -v ./output:/app/output \
    -e WEBHOOK_URL=$WEBHOOK_URL \
    -e WEBHOOK_API_KEY=$WEBHOOK_API_KEY \
    -e BACKEND_UPLOAD_URL=$BACKEND_UPLOAD_URL \
    -e BACKEND_UPLOAD_ADMIN_KEY=$BACKEND_UPLOAD_ADMIN_KEY \
    -e HF_HOME=/app/hf_cache \
    -e HUGGINGFACE_HUB_CACHE=/app/hf_cache \
    -e TRANSFORMERS_CACHE=/app/hf_cache \
    uncensored-tts:latest
