@echo off
echo Pulling latest images...
docker-compose -f docker-compose.yml pull
echo Update complete. Run start_services.cmd to apply.