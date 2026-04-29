import os
import time
import cv2
import numpy as np
import requests
from collections import defaultdict, deque
from ultralytics import YOLO
from datetime import datetime

# =========================
# Configuration
# =========================
MODEL_PATH      = os.getenv('YOLO_MODEL', 'yolov8n.pt')
VIDEO_SOURCE    = os.getenv('VIDEO_SOURCE', r'C:\Users\User\Downloads\Telegram Desktop\Run\4742451-uhd_3840_2160_25fps.mp4')
PERSON_CLASS_ID = 0
CONFIDENCE      = float(os.getenv('CONFIDENCE', '0.35'))
IMG_SIZE        = int(os.getenv('IMG_SIZE', '640'))
HALF_PRECISION  = os.getenv('HALF_PRECISION', 'true').lower() == 'true'

TRACKER           = os.getenv('TRACKER', 'botsort.yaml')
TRACK_HISTORY_SEC = float(os.getenv('TRACK_HISTORY_SEC', '1.5'))
MIN_TRACK_POINTS  = int(os.getenv('MIN_TRACK_POINTS', '4'))
FRAME_SKIP        = int(os.getenv('FRAME_SKIP', '1'))

CALIBRATE = os.getenv('CALIBRATE', 'false').lower() == 'true'

RUN_SPEED_NORM        = float(os.getenv('RUN_SPEED_NORM',        '1.5'))
RUN_CONFIRM_FRAMES    = int(os.getenv('RUN_CONFIRM_FRAMES',      '3'))
DIRECTION_CONSISTENCY = float(os.getenv('DIRECTION_CONSISTENCY', '0.3'))
EMA_ALPHA             = float(os.getenv('EMA_ALPHA',             '0.45'))

ALERT_COOLDOWN_SEC = int(os.getenv('ALERT_COOLDOWN_SEC', '30'))

USE_ZONE = os.getenv('USE_ZONE', 'false').lower() == 'true'
ZONE     = [(100, 100), (1180, 100), (1180, 650), (100, 650)]

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', "8548372776:AAEgz4mOaDBU8HyrCklRHFHsnxT9U7HtRA0")
TELEGRAM_CHAT_ID   = os.getenv('TELEGRAM_CHAT_ID',   "-5025882106")
SEND_SNAPSHOT      = os.getenv('SEND_SNAPSHOT', 'true').lower() == 'true'

SHOW_TRAIL       = True
TRAIL_THICKNESS  = 2
TRAIL_COLOR_RUN  = (0, 0, 255)
TRAIL_COLOR_WALK = (0, 255, 0)

# =========================
# New: Video Output Configuration
# =========================
SAVE_VIDEO_OUTPUT = True                    # Set to False if you don't want to save video
OUTPUT_FPS        = 25                      # You can change this
OUTPUT_CODEC      = 'mp4v'                  # 'mp4v' or 'XVID'

# =========================
# Helpers
# =========================
def parse_video_source(src):
    return int(src) if isinstance(src, str) and src.isdigit() else src

def point_in_zone(point, polygon):
    if len(polygon) < 3:
        return True
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def center_of_box(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def box_height(y1, y2):
    return max(abs(y2 - y1), 1)

def compute_raw_speed_px(history):
    pts = list(history)
    if len(pts) < 2:
        return 0.0
    total_dist, total_time = 0.0, 0.0
    for i in range(1, len(pts)):
        t1, p1, _ = pts[i - 1]
        t2, p2, _ = pts[i]
        dt = t2 - t1
        if dt < 1e-6:
            continue
        total_dist += np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        total_time += dt
    return (total_dist / total_time) if total_time > 1e-6 else 0.0

def compute_normalised_speed(history, bh):
    return compute_raw_speed_px(history) / bh

def compute_direction_consistency(history):
    pts = list(history)
    if len(pts) < 3:
        return 0.0
    vecs = []
    for i in range(1, len(pts)):
        _, p1, _ = pts[i - 1]
        _, p2, _ = pts[i]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        mag = np.hypot(dx, dy)
        if mag > 0.5:
            vecs.append((dx / mag, dy / mag))
    if len(vecs) < 2:
        return 0.0
    sims = [vecs[i-1][0]*vecs[i][0] + vecs[i-1][1]*vecs[i][1]
            for i in range(1, len(vecs))]
    return float(np.mean(sims))

def draw_zone(frame, polygon):
    if len(polygon) < 3:
        return
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (0, 255, 255), 3)

def send_telegram_message(text, image_path=None):
    if not TELEGRAM_BOT_TOKEN or 'YOUR_BOT_TOKEN' in TELEGRAM_BOT_TOKEN:
        return
    try:
        requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',
            data={'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': 'HTML'},
            timeout=8
        )
        if SEND_SNAPSHOT and image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as photo:
                requests.post(
                    f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto',
                    data={'chat_id': TELEGRAM_CHAT_ID, 'caption': '🚨 Running Person Detected'},
                    files={'photo': photo},
                    timeout=15
                )
    except Exception as e:
        print(f'[ERROR] Telegram: {e}')


# =========================
# Per-track state
# =========================
class TrackState:
    def __init__(self):
        self.history:      deque = deque(maxlen=90)
        self.smooth_speed: float = 0.0
        self.run_counter:  int   = 0
        self.is_running:   bool  = False
        self.max_speed:    float = 0.0
        self.max_dir:      float = -1.0

    def update(self, now, cx, cy, bh):
        self.history.append((now, (cx, cy), bh))
        while self.history and now - self.history[0][0] > TRACK_HISTORY_SEC:
            self.history.popleft()

        if len(self.history) < MIN_TRACK_POINTS:
            return 0.0, 0.0

        raw_speed   = compute_normalised_speed(self.history, bh)
        consistency = compute_direction_consistency(self.history)

        self.smooth_speed = EMA_ALPHA * raw_speed + (1.0 - EMA_ALPHA) * self.smooth_speed
        self.max_speed    = max(self.max_speed, self.smooth_speed)
        self.max_dir      = max(self.max_dir,   consistency)

        above = (self.smooth_speed >= RUN_SPEED_NORM and consistency >= DIRECTION_CONSISTENCY)

        self.run_counter = (min(self.run_counter + 1, RUN_CONFIRM_FRAMES + 5)
                            if above else max(self.run_counter - 1, 0))
        self.is_running  = self.run_counter >= RUN_CONFIRM_FRAMES
        return self.smooth_speed, consistency


# =========================
# Calibration table printer
# =========================
_calib_log: dict  = {}
_calib_t:   float = 0.0

def calib_update(tid, speed, direction):
    _calib_log[tid] = {'speed': speed, 'dir': direction}

def calib_print():
    global _calib_t
    now = time.time()
    if now - _calib_t < 2.0 or not _calib_log:
        return
    _calib_t = now
    print("\n[CALIBRATE] ── Live readings ───────────────────────────────────")
    print(f"  {'ID':>4}  {'Speed (bh/s)':>13}  {'Direction':>10}")
    print(f"  {'─'*4}  {'─'*13}  {'─'*10}")
    for tid, d in sorted(_calib_log.items()):
        print(f"  {tid:>4}  {d['speed']:>13.2f}  {d['dir']:>10.2f}")
    print(f"\n  Current threshold: speed >= {RUN_SPEED_NORM:.2f}  dir >= {DIRECTION_CONSISTENCY:.2f}")
    print("  → Observe speed values while people run.")
    print("  → Set RUN_SPEED_NORM = (running speed) * 0.75, then CALIBRATE=false")
    print("────────────────────────────────────────────────────────────────")


# =========================
# Main
# =========================
def main():
    source = parse_video_source(VIDEO_SOURCE)
    model  = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {VIDEO_SOURCE}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'[INFO] Input FPS: {fps:.1f}  Resolution: {width}x{height}  Source: {VIDEO_SOURCE}')

    # =========================
    # Video Writer Setup
    # =========================
    video_writer = None
    if SAVE_VIDEO_OUTPUT:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"running_detection_output_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
        video_writer = cv2.VideoWriter(output_filename, fourcc, OUTPUT_FPS, (width, height))
        
        print(f'[INFO] Video recording started → {output_filename}')

    if CALIBRATE:
        print('[CALIBRATE] Mode ON — no alerts will fire.')

    tracks:     dict[int, TrackState] = defaultdict(TrackState)
    last_alert: dict[int, float]      = {}
    frame_id    = 0
    prev_time   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if FRAME_SKIP > 1 and frame_id % FRAME_SKIP != 0:
            continue

        now     = time.time()
        display = frame.copy()
        if USE_ZONE:
            draw_zone(display, ZONE)

        results = model.track(
            frame, persist=True, tracker=TRACKER,
            classes=[PERSON_CLASS_ID], conf=CONFIDENCE,
            imgsz=IMG_SIZE, half=HALF_PRECISION, verbose=False,
        )

        running_count = 0
        active_ids: set[int] = set()

        if (results and results[0].boxes is not None and results[0].boxes.id is not None):
            boxes     = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist()
            xyxy      = boxes.xyxy.cpu().numpy()

            for track_id, box in zip(track_ids, xyxy):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = center_of_box(x1, y1, x2, y2)
                bh     = box_height(y1, y2)

                if USE_ZONE and not point_in_zone((cx, cy), ZONE):
                    continue

                active_ids.add(track_id)
                state = tracks[track_id]
                speed_norm, consistency = state.update(now, cx, cy, bh)

                is_running = state.is_running and not CALIBRATE
                color = TRAIL_COLOR_RUN if is_running else TRAIL_COLOR_WALK

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                raw_px = compute_raw_speed_px(state.history)
                lbl1 = f'ID:{track_id}  {speed_norm:.1f}bh/s  {raw_px:.0f}px/s'
                lbl2 = f'dir:{consistency:.2f}  {"RUNNING" if is_running else "walking"}'

                y0 = max(y1 - 22, 22)
                cv2.putText(display, lbl1, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
                cv2.putText(display, lbl2, (x1, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
                cv2.circle(display, (cx, cy), 4, color, -1)

                if SHOW_TRAIL and len(state.history) > 1:
                    pts = [p for _, p, _ in state.history]
                    for i in range(1, len(pts)):
                        cv2.line(display, pts[i-1], pts[i], color, TRAIL_THICKNESS)

                if CALIBRATE:
                    calib_update(track_id, speed_norm, consistency)

                if is_running:
                    running_count += 1
                    if now - last_alert.get(track_id, 0) >= ALERT_COOLDOWN_SEC:
                        snap = None
                        if SEND_SNAPSHOT:
                            snap = f"alert_{track_id}_{int(now)}.jpg"
                            cv2.imwrite(snap, display)
                        ts  = time.strftime('%Y-%m-%d %H:%M:%S')
                        msg = (f"🚨 <b>Running Person!</b>\n"
                               f"ID:{track_id}  {speed_norm:.1f}bh/s  dir:{consistency:.2f}\n"
                               f"{ts}")
                        send_telegram_message(msg, snap)
                        last_alert[track_id] = now

        # Clean stale tracks
        for oid in list(tracks.keys()):
            if (oid not in active_ids and 
                tracks[oid].history and 
                now - tracks[oid].history[-1][0] > TRACK_HISTORY_SEC * 3):
                del tracks[oid]
                _calib_log.pop(oid, None)

        if CALIBRATE:
            calib_print()

        # Write frame to output video
        if video_writer is not None:
            video_writer.write(display)

        fps_actual = 1.0 / (time.time() - prev_time + 1e-6)
        prev_time  = time.time()

        hud = 'CALIBRATE MODE' if CALIBRATE else f'Running: {running_count}'
        hud_color = (0, 165, 255) if CALIBRATE else (0, 215, 255)
        cv2.putText(display, hud, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, hud_color, 2)
        cv2.putText(display, f'FPS:{fps_actual:.1f}', (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(display,
                    f'thr: speed>={RUN_SPEED_NORM:.1f}bh/s  dir>={DIRECTION_CONSISTENCY:.2f}',
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow('Running Detection', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    print(f'[INFO] Video output saved as: running_detection_output_*.mp4')
    print('[INFO] Done.')


if __name__ == '__main__':
    main()