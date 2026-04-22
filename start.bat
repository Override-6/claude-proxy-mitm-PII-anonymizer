@echo off
echo === Claude Desktop MITM Proxy ===
echo.

:: Start mitmproxy with the capture addon
echo [1] Starting proxy on 127.0.0.1:8080 ...
start "mitmproxy" cmd /k "mitmdump -s src/proxy.py --listen-port 8080 --set console_eventlog_verbosity=info --set flow_detail=0"

:: Wait for proxy to be ready
timeout /t 18 /nobreak >nul

:: Start proxifier (WinDivert NAT) — requires Administrator
echo [2] Starting proxifier (captures all outbound :443 traffic)...
start "proxifier" cmd /k "sudo python proxifier.py"

timeout /t 3 /nobreak >nul

:: Launch Claude Desktop with cert bypass
echo [3] Launching Claude Desktop...
set NODE_TLS_REJECT_UNAUTHORIZED=0
for /f "delims=" %%i in ('powershell -Command "(Get-AppxPackage -Name 'Claude').InstallLocation"') do set CLAUDE_DIR=%%i
if not defined CLAUDE_DIR (
    echo ERROR: Claude Desktop package not found
    pause
    exit /b 1
)
start "" "%CLAUDE_DIR%\app\claude.exe" --proxy-server=127.0.0.1:8080


echo.
echo Proxy running on :8080, proxifier redirecting :443 traffic
echo Press any key to exit (proxy + proxifier keep running in their own windows)
pause >nul
