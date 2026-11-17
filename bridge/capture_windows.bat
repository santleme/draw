@echo off
rem ============================================================
rem  Draw - ponte de captura no Windows (rota reserva)
rem  Le a webcam pelo DirectShow e serve o video como MJPEG no
rem  adaptador host-only do WSL ^(porta 8080^) para o Draw no WSL.
rem
rem  Para conferir o nome exato da camera neste PC, rode:
rem    ffmpeg.exe -list_devices true -f dshow -i dummy
rem  O nome certo aparece na lista de dispositivos de video.
rem ============================================================

rem ===== CONFIGURACAO: nome da camera (edite se a lista mostrar outro nome) =====
set "CAM=1200W Camera"
rem ===============================================================================

set "FFMPEG=%~dp0ffmpeg.exe"

rem ===== ONDE ESCUTAR ==============================================
rem  LISTEN e o IP onde o ffmpeg escuta. O padrao resolve o IP do
rem  adaptador "vEthernet (WSL)": a rede virtual host-only entre
rem  Windows e WSL. Maquinas da rede local NAO alcancam esse
rem  endereco — so o WSL conecta, sem precisar de regra de firewall.
rem
rem  NAO use 0.0.0.0: serviria o video da webcam SEM SENHA para
rem  qualquer maquina que alcancasse o PC. Se outro bind for mesmo
rem  necessario, defina LISTEN explicitamente ciente da exposicao.
rem ===============================================================

if not defined LISTEN (
    for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Get-NetIPAddress -InterfaceAlias 'vEthernet (WSL*' -AddressFamily IPv4 -ErrorAction SilentlyContinue).IPAddress"`) do set "LISTEN=%%i"
)

if not defined LISTEN (
    echo [ERRO] adaptador "vEthernet (WSL)" nao encontrado.
    echo        Esta ponte existe para servir o video ao WSL; sem o
    echo        adaptador host-only do WSL nao ha bind seguro automatico.
    echo        Para expor na rede de proposito, defina LISTEN no .bat
    echo        ciente de que o video fica SEM SENHA na interface.
    pause
    exit /b 1
)

if not exist "%FFMPEG%" (
    echo [ERRO] ffmpeg.exe nao foi encontrado nesta pasta:
    echo        %~dp0
    echo.
    echo Baixe um build do ffmpeg para Windows:
    echo   - gyan.dev: https://www.gyan.dev/ffmpeg/builds/ ^(pacote release full^)
    echo   - ou os builds do BtbN no GitHub ^(ffmpeg-master-latest-win64-gpl^)
    echo.
    echo Extraia o ffmpeg.exe do arquivo baixado e coloque ele NESTA
    echo pasta, junto deste capture_windows.bat. Depois rode o .bat de novo.
    pause
    exit /b 1
)

echo Draw: servindo a camera "%CAM%" em http://%LISTEN%:8080/video
if "%LISTEN%"=="0.0.0.0" (
    echo [AVISO] 0.0.0.0 expoe o video SEM SENHA na rede local.
    echo         Volte para o padrao ^(adaptador WSL^) ou aceite o risco.
)
echo No WSL, conecte o Draw com o IP do Windows ^(nameserver em /etc/resolv.conf^):
echo   python3 main.py --source http://IP-DO-WINDOWS:8080/video
echo Feche esta janela para parar a ponte.
echo.

"%FFMPEG%" -f dshow -i video="%CAM%" -f mjpeg -q:v 5 -listen 1 http://%LISTEN%:8080/video

pause
