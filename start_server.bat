@echo off
rem PALEVO — запуск сервера. Использовать этот файл, а не "python main.py":
rem системный Python не имеет зависимостей проекта (uvicorn и т.д.),
rem они установлены в venv.
cd /d "%~dp0"
echo Запуск PALEVO... (http://localhost:8000)
venv\Scripts\python.exe main.py
pause
