@echo off
echo Enabling GPU support...
docker-compose -f docker-compose.yml up -d --force-recreate --build ollama
docker exec ollama ollama run llama2