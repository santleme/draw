#!/usr/bin/env python3
"""Serve um arquivo de video como MJPEG em tempo real (faca-camera).

Uso:
    python tools/mjpeg_server.py --video captures/sim.mp4 --port 8554 [--fps 15] [--loop]

O app consome com:  python main.py --source http://127.0.0.1:8554/video
"""

import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


def _open(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        cap.release()
        return None, None
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None, None
    return cap, frame


def make_handler(video, fps, loop, quality):
    interval = 1.0 / fps

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.rstrip("/") not in ("/video", ""):
                self.send_error(404)
                return
            # Sonda antes do 200: um vídeo que não abre ou não decoda
            # o primeiro frame vira erro na hora — nunca um worker
            # preso reabrindo o arquivo sem nunca escrever no socket.
            cap, frame = _open(video)
            if cap is None:
                self.send_error(502, "video unreadable")
                return
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while frame is not None:
                    while frame is not None:
                        t0 = time.monotonic()
                        ok, jpg = cv2.imencode(
                            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
                        )
                        if ok:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                + jpg.tobytes()
                                + b"\r\n"
                            )
                            self.wfile.flush()
                        ok, nxt = cap.read()
                        frame = nxt if ok else None
                        dt = time.monotonic() - t0
                        if dt < interval:
                            time.sleep(interval - dt)
                    if not loop:
                        return
                    # Só reinicia depois de uma passada que produziu
                    # frames — a sonda de _open garante isso; se a
                    # reabertura falhar, frame=None encerra o worker.
                    cap.release()
                    cap, frame = _open(video)
            except (BrokenPipeError, ConnectionResetError):
                return
            finally:
                if cap is not None:
                    cap.release()

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--loop", action="store_true", default=False)
    args = ap.parse_args()
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        make_handler(args.video, args.fps, args.loop, args.quality),
    )
    print(
        f"serving {args.video} at http://127.0.0.1:{args.port}/video @ {args.fps} fps"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
