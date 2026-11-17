@echo off
rem ============================================================
rem  Draw - ponte de captura no Windows (rota reserva)
rem  Le a webcam pelo DirectShow e serve o video como MJPEG em
rem  http://0.0.0.0:8080/video para o Draw rodando no WSL.
rem
rem  Para conferir o nome exato da camera neste PC, rode:
rem    ffmpeg.exe -list_devices true -f dshow -i dummy
rem  O nome certo aparece na lista de dispositivos de video.
rem ============================================================

rem ===== CONFIGURACAO: nome da camera (edite se a lista mostrar outro nome) =====
set "CAM=1200W Camera"
rem ===============================================================================

set "FFMPEG=%~dp0ffmpeg.exe"

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

echo Draw: servindo a camera "%CAM%" em http://0.0.0.0:8080/video
echo No WSL, conecte o Draw com o IP do Windows ^(nameserver em /etc/resolv.conf^):
echo   python3 main.py --source http://IP-DO-WINDOWS:8080/video
echo Feche esta janela para parar a ponte.
echo.

"%FFMPEG%" -f dshow -i video="%CAM%" -f mjpeg -q:v 5 -listen 1 http://0.0.0.0:8080/video

pause
