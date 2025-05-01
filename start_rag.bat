@echo off

:: Check if Docker Desktop is running
echo Checking if Docker Desktop is running...
docker ps >nul 2>nul
if %errorlevel%==0 (
    echo Docker Desktop is already running.
) else (
    echo Docker Desktop is not running. Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    
    echo Waiting for Docker Desktop to initialize...
    :docker_wait_loop
    timeout /t 5 /nobreak >nul
    docker ps >nul 2>nul
    if %errorlevel%==1 goto docker_wait_loop
    echo Docker Desktop is now running.
)

:: Launch Flask application
echo Starting Flask application...
start cmd /k "cd /d F:\AI builds exampls\RAG_chatboost && rg-venv\scripts\activate.bat && python app.py"

timeout /t 7 /nobreak >nul
:: Launch Chrome browser at http://127.0.0.1:5000
echo Opening Chrome browser...
start chrome.exe http://127.0.0.1:5000

:: Open a new terminal window and activate virtual environment
echo Opening terminal with virtual environment...
start cmd /k "cd /d F:\AI builds exampls\RAG_chatboost && rg-venv\scripts\activate.bat"

:: Open a new terminal window and start Ollama server
echo Starting Ollama server...
start cmd /k "ollama serve"

echo All applications started successfully.