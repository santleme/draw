# Windows helper protocol (v1)

Specification of the channel between the helper resident on Windows and Draw
running on WSL. This document is the contract: the reference receiver
is in [`../tools/helper_receiver.py`](../tools/helper_receiver.py) and
the helper sketch, in [`windows_helper.py`](windows_helper.py).

## Why this channel exists

Today the camera enters WSL via usbipd, and this route is fragile: the
Windows audio services (Audiosrv and AudioEndpointBuilder)
hold the camera's microphone function and attachment fails with
`Device busy`. An MJPEG stream over HTTP (ffmpeg on Windows serving the
camera) solves capture, but delivers images without frame identity
and without a timestamp: WSL receives pictures, not dated frames.

The approved design swaps both routes for a helper resident on
Windows that captures via DirectShow/MSMF, timestamps each frame **at
capture** and sends JPEG over a local socket. All processing and
all ML remain on WSL. No ML runs on Windows.

Without a capture timestamp there is no way to measure glass→ink latency, which
has never been measured end to end in this project. This protocol exists
to make that measurement possible (see "Step 0", below).

## Transport

- TCP, a single direction in v1: **helper → WSL**.
- The helper **listens** on a TCP port (default **8765**, configurable)
  and WSL **connects**. The helper is resident and outlives the app;
  the WSL side restarts frequently during development,
  so it is the one that initiates the connection.
- The helper binds to **`127.0.0.1` by default**. With WSL2 in
  *mirrored* networking mode, WSL reaches it as `127.0.0.1`. In the
  default *NAT* mode, bind the helper to the Windows-side IP of the
  `vEthernet (WSL)` adapter (`--host <ip>`); from WSL that IP is the
  `nameserver` of `/etc/resolv.conf`. A specific interface IP needs no
  extra flag.
- Binding to every interface (`0.0.0.0` / `::`) is refused unless the
  helper is started with `--allow-all-interfaces`, and then it prints
  a security warning. Do not expose the port outside the machine.
- The Windows firewall must allow the port only on the interface in
  use, as already happens with the ffmpeg bridge.
- No TLS in v1: the channel is local (Windows host ↔ WSL on the same
  PC). Every connection must pass the token handshake below before
  any frame is sent.

## Authentication handshake

Immediately after the TCP connection opens, and before the helper
sends any frame record, the WSL side sends a HELLO followed by the
shared token. All integers are little-endian.

| Offset | Size | Type   | Field       | Value |
|-------:|-----:|--------|-------------|-------|
| 0      | 4    | bytes  | `magic`     | `DRAH` (0x44 0x52 0x41 0x48) |
| 4      | 2    | uint16 | `version`   | 1 |
| 6      | 2    | uint16 | `token_len` | 16..256 |
| 8      | `token_len` | bytes | token | UTF-8 shared secret |

In Python, the HELLO is `struct.Struct("<4sHH")`, 8 bytes.

- The helper compares the token in constant time
  (`hmac.compare_digest`). On success it answers exactly the 4 bytes
  `DROK` and then starts the frame stream. On any failure (wrong
  magic or version, `token_len` out of range, wrong token, or no
  complete HELLO within 5 s) it closes the connection **without
  answering**, and keeps listening.
- The receiver treats anything other than `DROK` (including an
  early close) as an authentication error and reads no frames.
- The token comes from the `DRAW_HELPER_TOKEN` environment variable
  or from `--token-file` on both sides, and is never logged by the
  receiver. If the helper starts without one, it generates a random
  token (`secrets.token_urlsafe(32)`) and prints it once on its own
  console so the operator can copy it to WSL.
- The token authenticates the peer; it does not encrypt the stream.
  Frames still travel in clear, which is why the bind stays local.
- The handshake belongs to the socket only. Recorded files (below)
  start directly at the first frame record.

## Frame record

The stream is a sequence of records, without separators. Each record
is a **fixed-size binary header (28 bytes)** followed by the
JPEG payload. All integers are **little-endian**, with no padding
beyond that declared in the table.

| Offset | Size | Type   | Field           | Value in v1 |
|-------:|-----:|--------|-----------------|-------------|
| 0      | 4    | bytes  | `magic`         | `DRAW` (0x44 0x52 0x41 0x57) |
| 4      | 2    | uint16 | `version`       | 1 |
| 6      | 2    | uint16 | `reserved`      | 0 (ignore on reading) |
| 8      | 8    | uint64 | `frame_id`      | per-session counter, starts at 0 |
| 16     | 8    | int64  | `capture_ts_us` | microseconds, see anchor below |
| 24     | 4    | uint32 | `payload_len`   | size of the JPEG in bytes |
| 28     | `payload_len` | bytes | payload  | one complete JPEG frame |

In Python, the header is `struct.Struct("<4sHHQqI")`, 28 bytes.

Stream rules:

- `frame_id` is strictly increasing within a session and never
  reused in it. Gaps are possible and mean frames that
  capture discarded; the receiver **must not** block waiting for an id.
- The payload is exactly one self-contained JPEG (starts at `FF D8`,
  ends at `FF D9`). Resolution and quality are helper configuration,
  out of band in v1; the receiver discovers the resolution
  by decoding the JPEG.
- `payload_len` must be in `1..16 MiB`. The receiver checks it
  **before** reading or allocating the payload and fails with a
  protocol error otherwise, so a corrupt or hostile header cannot make
  it allocate an arbitrary buffer.
- A payload that does not start at `FF D8` and end at `FF D9`, whose
  SOF header is missing or truncated, whose dimensions exceed
  8192 px per side or 40 MP, or that the decoder rejects, is decoded
  as `None` by `decode_jpeg`; it never raises.
- There is no checksum in v1: TCP already guarantees integrity and order in
  transport, and the recorded file (below) is written and read on the same
  machine. A payload with an invalid `magic` in the following record
  indicates corruption or desynchronization; the v1 receiver **fails** with
  a protocol error, without trying to resynchronize.

## Timestamp anchor

`capture_ts_us` is the helper's clock at the moment closest that the
capture stack allows to the delivery of the frame to the helper, expressed in
microseconds over a **monotonic clock whose epoch is the start of the
helper process** (on Windows, `QueryPerformanceCounter`
converted; in Python, `time.perf_counter_ns() // 1000` has the same
semantics).

Caveats, which are part of the contract:

- A monotonic clock is not a wall clock: the value only serves for
  **differences within the same helper session** (intervals between
  frames, order, source jitter).
- **Clocks on different machines do not compare without a common
  anchor.** `t_receive_us - capture_ts_us` between WSL and Windows measures the
  offset between the two clocks, not latency. End-to-end latency
  is measured with the common anchor of step 0: a millisecond
  stopwatch filmed by the camera itself, whose reading travels
  inside the image.
- If the helper restarts, the epoch changes and `frame_id` returns to 0. A
  new session is a new epoch; series from different sessions do not
  mix without recording the change.

## Recorded file (replay)

The recording format is **the same sequence of records** that the
socket carries: a 28-byte header + payload, repeated, with no
file header and no footer. A recorded file is, byte for byte,
what passed through the socket.

Deliberate consequence: the receiver does not distinguish the source. The same
code that reads the socket reads the file, and any analysis (including
step 0) runs the same over the live session and over the replay.

- End of file in the middle of a record: the previous complete
  records are valid; the receiver reports the truncated tail as
  an error, not as a frame.
- On the socket, closing the connection ends the iteration in the same
  way as the end of the file.

## Step 0 (the measurement this protocol enables)

Protocol for measuring glass→ink latency, to run **twice on the
same yardstick**: once with the camera via usbipd (before) and once with the
helper (after). A millisecond stopwatch is filmed; for each
frame, `frame_id`, `t_capture_us` (source clock),
`t_receive_us` (WSL clock) and, when present, `t_screen_ms` (the
stopwatch reading inside that frame) are recorded. The tool is
[`../tools/step0_latency.py`](../tools/step0_latency.py).

The metric that decides is **jitter, p95 − median**, over **200+
frames**; the median alone does not decide. The screen reading error
(±16 ms at 60 Hz) is common to both runs and cancels out in the comparison.

## What v1 is and what is left for later

V1 is only this: **a one-way frame stream, helper → WSL**,
with the recorder and replay above. Explicitly outside v1:

- **Ack and flow control**: the helper sends at the
  capture cadence; if WSL lags, TCP holds the excess and the latency
  appears in step 0, it does not disappear. No frame is acknowledged.
- **Control channel WSL → helper**: start/stop, change
  resolution and quality, request state. Today this is
  manual helper configuration.
- **Cursor injection**: the helper sketch already declares injection
  via `SendInput` (normalized absolute coordinates with a virtual
  desktop), but transport of cursor commands **is not** part
  of this protocol in v1. It will be a channel of its own when the
  cursor bridge is built.
- **Clock anchor exchange** (the helper declaring its epoch in
  wall clock): it only enters if a real consumer needs
  to compare clocks between machines without the filmed stopwatch.
- **Alternative compression** (H.264, block JPEG): v1 is JPEG
  frame by frame, the same as the current bridge, so that replay is trivial.

Incompatible changes raise `version`. New fields enter first
in `reserved` or in a new header version, never changing the
meaning of an existing field.
