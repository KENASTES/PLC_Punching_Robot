import cv2
import time
import threading
import numpy as np

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


# ==========================================================
# Camera Settings
# ==========================================================

CAMERA_CONFIGS = [
    {"index": 0, "name": "Webcam 1 - Targets 1,2,3", "label_start": 1},
    {"index": 2, "name": "Webcam 2 - Targets 4,5,6", "label_start": 4},
]

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30


# ==========================================================
# Target Point / Color Detection Settings
# ==========================================================

POINTS_PER_CAMERA = 3
ROI_RADIUS = 30
MIN_COLOR_PIXELS = 45
MIN_BLOB_AREA = 15
SHOW_MASK = True


# ==========================================================
# Blue Baseline Calibration Settings
# ==========================================================

COLOR_SAMPLE_RADIUS = 5
H_TOL = 12
S_TOL = 80
V_TOL = 80
MIN_S_LOWER = 40
MIN_V_LOWER = 70


# ==========================================================
# ESP32 Serial Settings
# ==========================================================

SERIAL_PORT = None  # Example: "COM5". None = auto-detect first available port.
SERIAL_BANDWIDTH = 115200

SERIAL_POLL_SEC = 0.05
SERIAL_RECONNECT_SEC = 3.0
SERIAL_MODE_REPEAT_SEC = 1.0
WRITE_REPEAT_SEC = 0.25

MODE_WEBCAM = "WEB"
MODE_HUSKY = "HUSKY"


# ==========================================================
# UI Button Settings
# ==========================================================

BUTTON_X1 = 20
BUTTON_Y1 = 55
BUTTON_X2 = 285
BUTTON_Y2 = 95


# ==========================================================
# Program States
# ==========================================================

STATE_WAIT_COLOR = "WAIT_COLOR"
STATE_PICK_COLOR = "PICK_COLOR"
STATE_DETECT = "DETECT"


# ==========================================================
# Shared Variables
# ==========================================================

stop_event = threading.Event()
shared_lock = threading.Lock()

shared_detected_pos = None

selected_camera_mode = MODE_WEBCAM
esp_connected = False
esp_active_mode = "UNKNOWN"
esp_last_sent_pos = None
esp_last_sent_state = 0
esp_last_write_time = 0
esp_last_message = "ESP serial not started"

active_camera_context = None


# ==========================================================
# Threaded Camera Class
# ==========================================================

class CameraStream:
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print(f"[Error] Cannot open camera index {self.camera_index}.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def start(self):
        if not self.open():
            return False

        self.running = True
        self.thread = threading.Thread(target=self.update_loop, daemon=True)
        self.thread.start()
        return True

    def update_loop(self):
        while self.running and not stop_event.is_set():
            success, frame = self.cap.read()

            if success:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=1.0)

        if self.cap is not None:
            self.cap.release()


class CameraContext:
    def __init__(self, camera_index, window_name, label_start):
        self.camera_index = camera_index
        self.window_name = window_name
        self.label_start = label_start
        self.stream = CameraStream(camera_index)

        self.state = STATE_WAIT_COLOR
        self.target_points = []

        self.roi_masks = []
        self.roi_masks_shape = None
        self.roi_masks_dirty = True

        self.current_live_frame = None
        self.frozen_frame = None

        self.color_calibrated = False
        self.sampled_hsv = None
        self.hsv_lower = None
        self.hsv_upper = None

        self.last_detected_pos = None
        self.last_color_mask = None


# ==========================================================
# Drawing Helpers
# ==========================================================

def draw_transparent_text(
    frame,
    text,
    org,
    font_scale=0.55,
    text_color=(255, 255, 255),
    thickness=1,
    bg_color=(0, 0, 0),
    alpha=0.35,
    padding=6
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = org

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x1 = max(0, x - padding)
    y1 = max(0, y - text_h - baseline - padding)
    x2 = min(frame.shape[1], x + text_w + padding)
    y2 = min(frame.shape[0], y + baseline + padding)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def is_capture_button_clicked(x, y):
    return BUTTON_X1 <= x <= BUTTON_X2 and BUTTON_Y1 <= y <= BUTTON_Y2


def draw_capture_button(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (BUTTON_X1, BUTTON_Y1), (BUTTON_X2, BUTTON_Y2), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (BUTTON_X1, BUTTON_Y1), (BUTTON_X2, BUTTON_Y2), (255, 255, 255), 1)
    cv2.putText(
        frame,
        "Blue Baseline (C)",
        (BUTTON_X1 + 12, BUTTON_Y1 + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


def draw_minimal_status(ctx, frame):
    if ctx.state == STATE_WAIT_COLOR:
        status_text = f"{ctx.window_name}: Press C baseline | labels {ctx.label_start}-{ctx.label_start + 2}"
        status_color = (0, 255, 255)
    elif ctx.state == STATE_PICK_COLOR:
        status_text = f"{ctx.window_name}: Click blue baseline"
        status_color = (0, 255, 255)
    else:
        status_text = f"{ctx.window_name}: Click {POINTS_PER_CAMERA} points | W/H mode | Q quit"
        status_color = (0, 255, 0)

    draw_transparent_text(
        frame=frame,
        text=status_text,
        org=(20, 32),
        font_scale=0.50,
        text_color=status_color,
        thickness=1,
        bg_color=(0, 0, 0),
        alpha=0.35,
        padding=6
    )

    with shared_lock:
        selected_mode = selected_camera_mode
        connected = esp_connected
        active_mode = esp_active_mode
        last_message = esp_last_message
        detected_pos = shared_detected_pos

    if connected:
        esp_text = f"Selected={selected_mode} | ESP: Connected | Active={active_mode} | Out={detected_pos or 0} | {last_message}"
        esp_color = (0, 255, 0) if active_mode == MODE_WEBCAM else (0, 255, 255)
    else:
        esp_text = f"Selected={selected_mode} | ESP: Disconnected | Out={detected_pos or 0} | {last_message}"
        esp_color = (0, 0, 255)

    draw_transparent_text(
        frame=frame,
        text=esp_text,
        org=(20, frame.shape[0] - 15),
        font_scale=0.40,
        text_color=esp_color,
        thickness=1,
        bg_color=(0, 0, 0),
        alpha=0.30,
        padding=5
    )


# ==========================================================
# Blue Baseline Capture / Calibration
# ==========================================================

def start_blue_baseline_capture(ctx):
    if ctx.current_live_frame is None:
        print(f"[Warning] {ctx.window_name}: No live frame available yet.")
        return

    ctx.frozen_frame = ctx.current_live_frame.copy()
    ctx.state = STATE_PICK_COLOR

    print(f"{ctx.window_name}: Image frozen.")
    print(f"{ctx.window_name}: Click one point on the actual BLUE target color.")


def calibrate_blue_from_click(ctx, frame, x, y):
    h_img, w_img = frame.shape[:2]

    x1 = max(0, x - COLOR_SAMPLE_RADIUS)
    x2 = min(w_img, x + COLOR_SAMPLE_RADIUS + 1)
    y1 = max(0, y - COLOR_SAMPLE_RADIUS)
    y2 = min(h_img, y + COLOR_SAMPLE_RADIUS + 1)

    patch = frame[y1:y2, x1:x2]

    if patch.size == 0:
        print(f"[Warning] {ctx.window_name}: Invalid color sample area.")
        return False

    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    pixels = hsv_patch.reshape(-1, 3)

    score = pixels[:, 1].astype(np.int32) + pixels[:, 2].astype(np.int32)
    top_count = max(5, len(pixels) // 3)
    top_pixels = pixels[np.argsort(score)[-top_count:]]
    median_hsv = np.median(top_pixels, axis=0).astype(int)

    h, s, v = int(median_hsv[0]), int(median_hsv[1]), int(median_hsv[2])

    h_low = max(0, h - H_TOL)
    h_high = min(179, h + H_TOL)
    s_low = max(MIN_S_LOWER, s - S_TOL)
    v_low = max(MIN_V_LOWER, v - V_TOL)

    ctx.hsv_lower = np.array([h_low, s_low, v_low])
    ctx.hsv_upper = np.array([h_high, 255, 255])
    ctx.sampled_hsv = (h, s, v)
    ctx.color_calibrated = True

    print(f"{ctx.window_name}: Blue baseline calibrated.")
    print(f"{ctx.window_name}: Sampled HSV H={h}, S={s}, V={v}")
    print(f"{ctx.window_name}: HSV lower {ctx.hsv_lower.tolist()}")
    print(f"{ctx.window_name}: HSV upper {ctx.hsv_upper.tolist()}")

    if s < 50 or v < 80:
        print(f"[Warning] {ctx.window_name}: Selected point may not be bright/saturated enough.")

    return True


# ==========================================================
# ROI Mask Helper
# ==========================================================

def rebuild_roi_masks(ctx, frame_shape):
    h, w = frame_shape[:2]
    ctx.roi_masks = []

    for pt in ctx.target_points:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, pt, ROI_RADIUS, 255, -1)
        ctx.roi_masks.append(mask)

    ctx.roi_masks_shape = (h, w)
    ctx.roi_masks_dirty = False


def reset_camera_points(ctx, reason):
    ctx.target_points = []
    ctx.roi_masks_dirty = True
    ctx.last_detected_pos = None

    with shared_lock:
        global shared_detected_pos
        shared_detected_pos = None

    print(f"{ctx.window_name}: Target points reset by {reason}.")


# ==========================================================
# Mouse Callback
# ==========================================================

def mouse_callback(event, x, y, flags, ctx):
    global active_camera_context

    active_camera_context = ctx

    if event == cv2.EVENT_LBUTTONDOWN:
        if ctx.state == STATE_PICK_COLOR:
            if ctx.frozen_frame is None:
                print(f"[Warning] {ctx.window_name}: No frozen frame available.")
                return

            success = calibrate_blue_from_click(ctx, ctx.frozen_frame, x, y)

            if success:
                ctx.state = STATE_DETECT
                ctx.frozen_frame = None
                print(f"{ctx.window_name}: Live image resumed.")
                print(f"{ctx.window_name}: Click target positions {ctx.label_start}-{ctx.label_start + 2}.")

            return

        if ctx.state == STATE_WAIT_COLOR and is_capture_button_clicked(x, y):
            start_blue_baseline_capture(ctx)
            return

        if ctx.state == STATE_WAIT_COLOR:
            print(f"{ctx.window_name}: Please capture and select the blue baseline first.")
            return

        if ctx.state == STATE_DETECT:
            if len(ctx.target_points) < POINTS_PER_CAMERA:
                ctx.target_points.append((x, y))
                ctx.roi_masks_dirty = True

                label = ctx.label_start + len(ctx.target_points) - 1
                print(f"{ctx.window_name}: Set Position {label} at {(x, y)}")
            else:
                print(f"{ctx.window_name}: Maximum {POINTS_PER_CAMERA} points already set. Press R to reset all.")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if ctx.state == STATE_PICK_COLOR:
            ctx.frozen_frame = None
            ctx.state = STATE_DETECT if ctx.color_calibrated else STATE_WAIT_COLOR
            print(f"{ctx.window_name}: Blue baseline selection cancelled.")
            return

        reset_camera_points(ctx, "right-click")


# ==========================================================
# ESP Serial Communication
# ==========================================================

def find_esp_serial_port():
    if SERIAL_PORT:
        return SERIAL_PORT

    if list_ports is None:
        return None

    ports = list(list_ports.comports())

    preferred_keywords = (
        "USB",
        "CH340",
        "CP210",
        "Silicon Labs",
        "UART",
        "ESP",
    )

    for port in ports:
        text = f"{port.device} {port.description} {port.manufacturer or ''}"
        if any(keyword.lower() in text.lower() for keyword in preferred_keywords):
            return port.device

    if ports:
        return ports[0].device

    return None


def parse_esp_line(line):
    global esp_active_mode, esp_last_message

    clean_line = line.strip()
    if not clean_line:
        return

    with shared_lock:
        if clean_line.startswith("MODE,"):
            parts = clean_line.split(",")
            esp_active_mode = parts[1].strip() if len(parts) > 1 else "UNKNOWN"
            esp_last_message = clean_line
        elif clean_line.startswith("ACK,"):
            esp_last_message = clean_line
        else:
            esp_last_message = clean_line[-80:]


def set_camera_mode(mode):
    global selected_camera_mode

    if mode not in (MODE_WEBCAM, MODE_HUSKY):
        return

    with shared_lock:
        selected_camera_mode = mode

    print(f"Camera mode selected: {mode}")


def normalize_key(key):
    if key < 0:
        return ""

    ascii_key = key & 0xFF

    if 0 <= ascii_key <= 255:
        try:
            return chr(ascii_key).lower()
        except ValueError:
            return ""

    return ""


def serial_worker():
    global esp_connected, esp_active_mode
    global esp_last_sent_pos, esp_last_sent_state, esp_last_write_time
    global esp_last_message

    if serial is None:
        with shared_lock:
            esp_connected = False
            esp_last_message = "pyserial missing. Install with: pip install pyserial"
        return

    ser = None
    last_reconnect_attempt = 0
    last_mode_write_time = 0

    while not stop_event.is_set():
        if ser is None or not ser.is_open:
            now = time.time()
            if now - last_reconnect_attempt < SERIAL_RECONNECT_SEC:
                time.sleep(0.05)
                continue

            last_reconnect_attempt = now
            port_name = find_esp_serial_port()

            if port_name is None:
                with shared_lock:
                    esp_connected = False
                    esp_active_mode = "UNKNOWN"
                    esp_last_message = "No serial port found"
                continue

            try:
                ser = serial.Serial(
                    port=port_name,
                    baudrate=SERIAL_BANDWIDTH,
                    timeout=0,
                    write_timeout=0.1,
                )

                time.sleep(2.0)
                ser.reset_input_buffer()
                ser.reset_output_buffer()

                with shared_lock:
                    esp_connected = True
                    esp_active_mode = "UNKNOWN"
                    esp_last_message = f"Port {port_name}"
                    esp_last_sent_pos = None
                    esp_last_sent_state = 0
                    esp_last_write_time = 0

                print(f"ESP serial connected on {port_name} @ {SERIAL_BANDWIDTH}.")

            except Exception as exc:
                error_text = str(exc)
                if isinstance(exc, PermissionError) or "PermissionError" in error_text or "Access is denied" in error_text:
                    error_text = f"Port {port_name} busy/access denied. Close Arduino Serial Monitor or other serial apps."

                with shared_lock:
                    esp_connected = False
                    esp_active_mode = "UNKNOWN"
                    esp_last_message = error_text

                try:
                    if ser is not None:
                        ser.close()
                except Exception:
                    pass

                ser = None
                continue

        try:
            while ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="replace")
                parse_esp_line(line)

            now = time.time()

            with shared_lock:
                mode = selected_camera_mode
                detected_pos = shared_detected_pos
                last_pos = esp_last_sent_pos
                last_state = esp_last_sent_state
                last_write = esp_last_write_time

            if now - last_mode_write_time >= SERIAL_MODE_REPEAT_SEC:
                ser.write(f"MODE,{mode}\n".encode("ascii"))
                last_mode_write_time = now

            target_pos = int(detected_pos) if detected_pos is not None else 0
            box_state = 1 if target_pos > 0 else 0

            should_send_data = (
                mode == MODE_WEBCAM and
                (
                    target_pos != last_pos or
                    box_state != last_state or
                    now - last_write >= WRITE_REPEAT_SEC
                )
            )

            if should_send_data:
                ser.write(f"DATA,{target_pos},{box_state}\n".encode("ascii"))

                with shared_lock:
                    esp_last_sent_pos = target_pos
                    esp_last_sent_state = box_state
                    esp_last_write_time = now
                    esp_last_message = f"Sent DATA,{target_pos},{box_state}"

        except Exception as exc:
            with shared_lock:
                esp_connected = False
                esp_active_mode = "UNKNOWN"
                esp_last_message = f"Serial error: {exc}"

            print(f"[Warning] ESP serial disconnected: {exc}")

            try:
                ser.close()
            except Exception:
                pass

            ser = None

        time.sleep(SERIAL_POLL_SEC)

    if ser is not None:
        try:
            ser.write(f"MODE,{MODE_HUSKY}\n".encode("ascii"))
            ser.close()
        except Exception:
            pass


# ==========================================================
# Color-Based Target Detection
# ==========================================================

def detect_calibrated_target_position(ctx, frame):
    if ctx.hsv_lower is None or ctx.hsv_upper is None:
        return None, np.zeros(frame.shape[:2], dtype=np.uint8), []

    if len(ctx.target_points) == 0:
        return None, np.zeros(frame.shape[:2], dtype=np.uint8), []

    h, w = frame.shape[:2]

    if ctx.roi_masks_dirty or ctx.roi_masks_shape != (h, w):
        rebuild_roi_masks(ctx, frame.shape)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, ctx.hsv_lower, ctx.hsv_upper)

    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.dilate(color_mask, kernel, iterations=1)

    best_pos = None
    best_score = 0
    scores = []

    for i, mask in enumerate(ctx.roi_masks):
        roi_color = cv2.bitwise_and(color_mask, color_mask, mask=mask)
        color_pixels = cv2.countNonZero(roi_color)

        max_area = 0

        if color_pixels >= MIN_COLOR_PIXELS:
            contours, _ = cv2.findContours(
                roi_color,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for c in contours:
                area = cv2.contourArea(c)
                if area > max_area:
                    max_area = area

        scores.append((color_pixels, max_area))

        if color_pixels >= MIN_COLOR_PIXELS and max_area >= MIN_BLOB_AREA:
            if color_pixels > best_score:
                best_score = color_pixels
                best_pos = ctx.label_start + i

    return best_pos, color_mask, scores


def read_camera_frame(ctx):
    if ctx.state == STATE_PICK_COLOR and ctx.frozen_frame is not None:
        return ctx.frozen_frame.copy()

    live_frame = ctx.stream.read()

    if live_frame is None:
        return None

    ctx.current_live_frame = live_frame.copy()
    return live_frame.copy()


def process_camera(ctx):
    frame = read_camera_frame(ctx)

    if frame is None:
        return None, None

    detected_pos = None
    color_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if ctx.state == STATE_DETECT and ctx.color_calibrated:
        detected_pos, color_mask, _ = detect_calibrated_target_position(ctx, frame)

    ctx.last_color_mask = color_mask

    for i, pt in enumerate(ctx.target_points):
        x, y = pt
        global_label = ctx.label_start + i
        color = (255, 0, 0) if detected_pos == global_label else (0, 255, 0)

        cv2.circle(frame, pt, 5, color, -1)
        cv2.circle(frame, pt, ROI_RADIUS, color, 2)

        cv2.putText(
            frame,
            f"P{global_label}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA
        )

    if ctx.state == STATE_DETECT and detected_pos is not None:
        if detected_pos != ctx.last_detected_pos:
            print(f"{ctx.window_name}: Calibrated blue target detected at Position {detected_pos}")
            ctx.last_detected_pos = detected_pos
    else:
        ctx.last_detected_pos = None

    draw_minimal_status(ctx, frame)

    if ctx.state == STATE_WAIT_COLOR:
        draw_capture_button(frame)

    return frame, detected_pos


def combine_detected_positions(detected_positions):
    for detected_pos in detected_positions:
        if detected_pos is not None:
            return detected_pos
    return None


def reset_all_points(camera_contexts):
    global shared_detected_pos

    for ctx in camera_contexts:
        ctx.target_points = []
        ctx.roi_masks_dirty = True
        ctx.last_detected_pos = None

    with shared_lock:
        shared_detected_pos = None

    print("All target points reset. Please click points again.")


# ==========================================================
# Main Program
# ==========================================================

camera_contexts = [
    CameraContext(
        camera_index=config["index"],
        window_name=config["name"],
        label_start=config["label_start"]
    )
    for config in CAMERA_CONFIGS
]

started_contexts = []

for ctx in camera_contexts:
    if not ctx.stream.start():
        stop_event.set()
        for started_ctx in started_contexts:
            started_ctx.stream.stop()
        raise SystemExit(1)

    started_contexts.append(ctx)

serial_thread = threading.Thread(target=serial_worker, daemon=True)
serial_thread.start()

for ctx in camera_contexts:
    cv2.namedWindow(ctx.window_name)
    cv2.setMouseCallback(ctx.window_name, mouse_callback, ctx)

print("Double webcam program started.")
print("Output path is unchanged: PC Serial -> ESP32 using MODE/DATA protocol.")
print("Webcam 1 labels: 1, 2, 3")
print("Webcam 2 labels: 4, 5, 6")
print("Controls:")
print("- c/C = capture still image for blue baseline on the last clicked webcam window")
print("- r/R = reset target points on both webcams")
print("- m/M = show/hide calibrated color masks")
print("- w/W = Webcam mode (PC serial to ESP32)")
print("- h/H = Husky mode (ESP32 fallback camera)")
print("- q/Q = quit")
print("- right-click = reset target points on clicked webcam, or cancel baseline selection")
print(f"ESP serial protocol: MODE,<WEB|HUSKY> and DATA,<target 0-6>,<state 0-1> @ {SERIAL_BANDWIDTH}.")

try:
    while True:
        frames = []
        detected_positions = []

        for ctx in camera_contexts:
            frame, detected_pos = process_camera(ctx)
            frames.append((ctx, frame))
            detected_positions.append(detected_pos)

        combined_detected_pos = combine_detected_positions(detected_positions)

        with shared_lock:
            shared_detected_pos = combined_detected_pos

        for ctx, frame in frames:
            if frame is None:
                continue

            cv2.imshow(ctx.window_name, frame)

            mask_window_name = f"Mask - {ctx.window_name}"
            if SHOW_MASK and ctx.last_color_mask is not None:
                cv2.imshow(mask_window_name, ctx.last_color_mask)
            else:
                try:
                    cv2.destroyWindow(mask_window_name)
                except cv2.error:
                    pass

        key_code = cv2.waitKeyEx(1)
        key = normalize_key(key_code)

        if key == "q":
            break

        elif key == "c":
            target_ctx = active_camera_context

            if target_ctx is None:
                pending_contexts = [
                    ctx for ctx in camera_contexts
                    if ctx.state != STATE_PICK_COLOR and not ctx.color_calibrated
                ]
                target_ctx = pending_contexts[0] if pending_contexts else camera_contexts[0]

            if target_ctx.state != STATE_PICK_COLOR:
                start_blue_baseline_capture(target_ctx)
            else:
                print(f"{target_ctx.window_name}: Already selecting blue baseline.")

        elif key == "r":
            reset_all_points(camera_contexts)

        elif key == "m":
            SHOW_MASK = not SHOW_MASK
            print(f"SHOW_MASK = {SHOW_MASK}")

        elif key == "w":
            set_camera_mode(MODE_WEBCAM)

        elif key == "h":
            set_camera_mode(MODE_HUSKY)

        elif key_code == 27:
            for ctx in camera_contexts:
                if ctx.state == STATE_PICK_COLOR:
                    ctx.frozen_frame = None
                    ctx.state = STATE_DETECT if ctx.color_calibrated else STATE_WAIT_COLOR
                    print(f"{ctx.window_name}: Blue baseline selection cancelled.")

finally:
    stop_event.set()

    for ctx in camera_contexts:
        ctx.stream.stop()

    serial_thread.join(timeout=1.0)
    cv2.destroyAllWindows()
