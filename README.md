# 🏃 Running Person Detection System

A real-time computer vision system that detects running people using YOLOv8 object tracking, with Telegram alerts and video output recording.

---

## 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Calibration Mode](#calibration-mode)
- [Telegram Alerts](#telegram-alerts)
- [Zone Detection](#zone-detection)
- [Output](#output)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- 🎯 Real-time person detection and tracking using **YOLOv8 + BotSORT**
- 🏃 Run/walk classification based on normalised speed and direction consistency
- 📍 Optional **zone-based** detection (only alert within a defined polygon)
- 📲 **Telegram bot alerts** with snapshot images
- 🎥 **Video output recording** with annotated bounding boxes and trails
- 🔧 **Calibration mode** to tune thresholds for your specific camera/environment
- 📊 Per-track HUD showing speed, direction, and status

---

## 🖥️ Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for real-time performance)

### Python Dependencies

```bash
pip install ultralytics opencv-python numpy requests
```

Or install all at once:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
requests>=2.28.0
```

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/running-detection.git
cd running-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. YOLOv8 model will auto-download on first run
#    Or manually place your model at the path set by YOLO_MODEL
```

---

## ⚙️ Configuration

All settings can be configured via **environment variables** or by editing the constants at the top of the script.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | Path to YOLO model file |
| `VIDEO_SOURCE` | *(path to video)* | Video file path or camera index (e.g. `0`) |
| `CONFIDENCE` | `0.35` | Minimum detection confidence (0.0–1.0) |
| `IMG_SIZE` | `640` | YOLO inference image size |
| `HALF_PRECISION` | `true` | Use FP16 for faster GPU inference |

### Tracking Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRACKER` | `botsort.yaml` | Tracker config (`botsort.yaml` or `bytetrack.yaml`) |
| `TRACK_HISTORY_SEC` | `1.5` | Seconds of track history to keep |
| `MIN_TRACK_POINTS` | `4` | Minimum points before speed is computed |
| `FRAME_SKIP` | `1` | Process every Nth frame (1 = all frames) |

### Run Detection Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_SPEED_NORM` | `1.5` | Speed threshold in box-heights/second |
| `RUN_CONFIRM_FRAMES` | `3` | Frames above threshold before labelling as running |
| `DIRECTION_CONSISTENCY` | `0.3` | Minimum directional consistency (-1.0 to 1.0) |
| `EMA_ALPHA` | `0.45` | Exponential moving average smoothing factor |

### Alert Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ALERT_COOLDOWN_SEC` | `30` | Seconds between repeated alerts for the same person |
| `TELEGRAM_BOT_TOKEN` | *(your token)* | Telegram bot token |
| `TELEGRAM_CHAT_ID` | *(your chat ID)* | Telegram chat or group ID |
| `SEND_SNAPSHOT` | `true` | Attach a snapshot image to Telegram alerts |

### Zone Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_ZONE` | `false` | Only detect running inside the defined polygon |
| `ZONE` | Rectangle | List of `(x, y)` polygon vertices |

### Calibration

| Variable | Default | Description |
|----------|---------|-------------|
| `CALIBRATE` | `false` | Enable calibration mode (no alerts, shows live readings) |

---

## ▶️ Usage

### Run with default settings

```bash
python running_detection.py
```

### Run with a different video source

```bash
VIDEO_SOURCE=0 python running_detection.py            # Webcam
VIDEO_SOURCE=/path/to/video.mp4 python running_detection.py
```

### Run with custom thresholds

```bash
RUN_SPEED_NORM=2.0 DIRECTION_CONSISTENCY=0.4 python running_detection.py
```

### Run with zone detection enabled

```bash
USE_ZONE=true python running_detection.py
```

### Keyboard Shortcut

| Key | Action |
|-----|--------|
| `Q` | Quit the application |

---

## 🔧 Calibration Mode

Use calibration mode to find the right `RUN_SPEED_NORM` threshold for your camera setup:

```bash
CALIBRATE=true python running_detection.py
```

While running, the console prints a live table every 2 seconds:

```
[CALIBRATE] ── Live readings ───────────────────────────────────
    ID   Speed (bh/s)   Direction
  ────  ─────────────  ──────────
     1          2.31        0.87
     2          0.54        0.21

  Current threshold: speed >= 1.50  dir >= 0.30
  → Observe speed values while people run.
  → Set RUN_SPEED_NORM = (running speed) * 0.75, then CALIBRATE=false
────────────────────────────────────────────────────────────────
```

**Steps:**
1. Set `CALIBRATE=true` and run the script
2. Walk and run in front of the camera
3. Note the speed values for running vs walking
4. Set `RUN_SPEED_NORM` to approximately **75% of the observed running speed**
5. Set `CALIBRATE=false` and restart

---

## 📲 Telegram Alerts

### Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) and copy the token
2. Get your chat ID (send a message to the bot and check `https://api.telegram.org/bot<TOKEN>/getUpdates`)
3. Set the environment variables:

```bash
export TELEGRAM_BOT_TOKEN="your_bot_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"
```

### Alert Format

When a running person is detected, you receive:

```
🚨 Running Person!
ID:3  2.1bh/s  dir:0.87
2025-01-15 14:32:01
```

With an optional snapshot image attached (controlled by `SEND_SNAPSHOT`).

---

## 📐 Zone Detection

Define a polygon to restrict detection to a specific area of the frame:

```python
# In the script, edit the ZONE variable:
ZONE = [(100, 100), (1180, 100), (1180, 650), (100, 650)]
# Format: list of (x, y) pixel coordinate tuples
```

Enable with:
```bash
USE_ZONE=true python running_detection.py
```

The zone boundary is drawn in **cyan** on the display. Only persons whose bounding box center falls within the polygon will be tracked and alerted on.

---

## 🎥 Output

### Live Display

The annotated video window shows:

- **Green boxes** — walking persons
- **Red boxes** — running persons
- **Trail lines** — recent movement history
- **HUD labels** — track ID, speed (bh/s), pixel speed (px/s), direction score, and status
- **Top-left overlay** — running person count, FPS, and active thresholds

### Saved Video

By default, an annotated video is saved to the working directory:

```
running_detection_output_YYYYMMDD_HHMMSS.mp4
```

To disable saving:
```python
SAVE_VIDEO_OUTPUT = False   # in the script
```

| Setting | Value |
|---------|-------|
| `SAVE_VIDEO_OUTPUT` | `True` |
| `OUTPUT_FPS` | `25` |
| `OUTPUT_CODEC` | `mp4v` |

---

## 🧠 How It Works

### Speed Metric: Box-Heights per Second (bh/s)

Speed is normalised by the person's bounding box height to be **camera-distance-independent**. A person running at 2.0 bh/s appears to move at roughly the same rate whether they are close or far from the camera.

```
normalised_speed = pixel_speed (px/s) / box_height (px)
```

### Direction Consistency

Measures how consistently a person moves in the same direction across recent frames. Values close to `1.0` mean very straight motion; values near `0.0` or below suggest erratic movement.

### Confirmation Logic

A person is only labelled **RUNNING** after `RUN_CONFIRM_FRAMES` consecutive frames above both the speed and direction thresholds. This eliminates false positives from brief fast movements.

### EMA Smoothing

Raw speed values are smoothed using an **exponential moving average** to reduce noise:

```
smooth_speed = α × raw_speed + (1 - α) × smooth_speed
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| No detections | Lower `CONFIDENCE` (try `0.25`) |
| Too many false positives | Raise `RUN_SPEED_NORM` or `RUN_CONFIRM_FRAMES` |
| Running not detected | Lower `RUN_SPEED_NORM` — use calibration mode |
| Laggy performance | Increase `FRAME_SKIP`, use `yolov8n.pt`, enable `HALF_PRECISION=true` |
| Telegram alerts not sending | Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`; check bot is added to the chat |
| Video file not opening | Check `VIDEO_SOURCE` path; for webcam use integer index (e.g. `0`) |
| Tracks flickering | Lower `CONFIDENCE`, or switch tracker to `bytetrack.yaml` |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [BotSORT Tracker](https://github.com/NirAharon/BoT-SORT)
