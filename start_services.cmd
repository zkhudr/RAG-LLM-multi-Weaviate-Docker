@echo off
echo Starting Weaviate and Ollama...
docker-compose -f docker-compose.yml up -d --build
timeout /t 15
curl http://localhost:8080/v1/meta

pause

@echo off
echo Active Containers:
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | findstr "weaviate ollama"

echo Weaviate Logs:
docker logs --tail=20 weaviate

echo Ollama Logs:
docker logs --tail=20 ollama

pause
@echo off
echo Testing Weaviate...
curl -s -o nul -w "Weaviate HTTP Status: %%{http_code}\n" http://localhost:8080/v1/meta

echo Testing Ollama...
curl -s -o nul -w "Ollama HTTP Status: %%{http_code}\n" http://localhost:11434/api/tags

echo Testing Integration...
curl -X POST http://localhost:8080/v1/modules/text2vec-ollama/configure -H "Content-Type: application/json" -d "{\"ollamaEndpoint\": \"http://ollama:11434\"}"
pause
