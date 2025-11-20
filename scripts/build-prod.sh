docker buildx create --name builder
docker buildx use builder
docker buildx build --push --platform linux/arm64,linux/amd64 -t rein1605/uncensored-tts:v1.0.0 . -f Dockerfile
