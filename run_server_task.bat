@echo off
rem PALEVO — самовосстанавливающаяся обёртка сервера.
rem Цикл: если порт 8000 свободен — запускаем сервер; при падении —
rem перезапуск через 5 с. Несколько копий безопасны: пока порт занят,
rem лишняя копия просто ждёт.
rem Автозапуск при входе в Windows — ярлык на этот файл в
rem   shell:startup  (не требует прав администратора).
rem Вариант без окна консоли (нужен админ):
rem   schtasks /Create /TN "PALEVO_Server" /TR "\"%~f0\"" /SC ONLOGON /F
cd /d "%~dp0"
:loop
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
  rem порт занят — сервер уже запущен (другой копией), ждём
  timeout /t 30 /nobreak >nul
  goto loop
)
echo === server-restart %date% %time% === >> data\app_run.log
venv\Scripts\python.exe main.py >> data\app_run.log 2>&1
echo === exit code %errorlevel%, перезапуск через 5 с === >> data\app_run.log
timeout /t 5 /nobreak >nul
goto loop
