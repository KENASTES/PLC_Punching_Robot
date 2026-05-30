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

CAMERA_INDEX = 0

# Lower resolution improves FPS.
# If you need higher precision, try 1280 x 720.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30


# ==========================================================
# Target Point / Color Detection Settings
# ==========================================================

MAX_POINTS = 6

# Radius around each clicked point to check for calibrated blue color
ROI_RADIUS = 30

# Minimum detected pixels inside ROI to confirm detection
MIN_COLOR_PIXELS = 45

# Minimum blob area inside ROI
MIN_BLOB_AREA = 15

# Optional mask debug window. Default is hidden.
SHOW_MASK = True


# ==========================================================
# Blue Baseline Calibration Settings
# ==========================================================

# Patch radius around the clicked blue baseline point
COLOR_SAMPLE_RADIUS = 5

# HSV tolerance around selected blue baseline
H_TOL = 12
S_TOL = 80
V_TOL = 80

# Safety lower limits for saturation and brightness
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

# Prevent repeated write every frame.
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

state = STATE_WAIT_COLOR


# ==========================================================
# Global Variables
# ==========================================================

target_points = []

roi_masks = []
roi_masks_shape = None
roi_masks_dirty = True

current_live_frame = None
frozen_frame = None

color_calibrated = False
sampled_hsv = None

HSV_LOWER = None
HSV_UPPER = None

last_detected_pos = None

stop_event = threading.Event()

# Shared detected target for ESP serial thread
shared_detected_pos = None

# Shared ESP serial status
selected_camera_mode = MODE_WEBCAM
esp_connected = False
esp_active_mode = "UNKNOWN"
esp_last_sent_pos = None
esp_last_sent_state = 0
esp_last_write_time = 0
esp_last_message = "ESP serial not started"

shared_lock = threading.Lock()


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
        # CAP_DSHOW can reduce latency on Windows.
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print(f"[Warning] Cannot open camera index {self.camera_index}. Trying camera index 0...")
            self.cap.release()
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print("[Error] Cannot open camera.")
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


# ==========================================================
# Transparent Text Helper
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
    """
    Draw small text with a semi-transparent background.
    org is the normal cv2.putText baseline position.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = org

    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    x1 = max(0, x - padding)
    y1 = max(0, y - text_h - baseline - padding)
    x2 = min(frame.shape[1], x + text_w + padding)
    y2 = min(frame.shape[0], y + baseline + padding)

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        bg_color,
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )


# ==========================================================
# UI Helper Functions
# ==========================================================

def is_capture_button_clicked(x, y):
    return BUTTON_X1 <= x <= BUTTON_X2 and BUTTON_Y1 <= y <= BUTTON_Y2


def draw_capture_button(frame):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (BUTTON_X1, BUTTON_Y1),
        (BUTTON_X2, BUTTON_Y2),
        (40, 40, 40),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.45,
        frame,
        0.55,
        0,
        frame
    )

    cv2.rectangle(
        frame,
        (BUTTON_X1, BUTTON_Y1),
        (BUTTON_X2, BUTTON_Y2),
        (255, 255, 255),
        1
    )

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


def draw_minimal_status(frame):
    """
    Clean on-image UI:
    - Small instruction text
    - Small ESP serial status text
    - Transparent background
    """

    if state == STATE_WAIT_COLOR:
        status_text = "Press C for Blue Baseline | W/H mode | Q quit"
        status_color = (0, 255, 255)

    elif state == STATE_PICK_COLOR:
        status_text = "Click Blue Color Baseline"
        status_color = (0, 255, 255)

    else:
        status_text = "Click 6 points"
        status_color = (0, 255, 0)

    # Main instruction text with transparent background
    draw_transparent_text(
        frame=frame,
        text=status_text,
        org=(20, 32),
        font_scale=0.55,
        text_color=status_color,
        thickness=1,
        bg_color=(0, 0, 0),
        alpha=0.35,
        padding=6
    )

    # Small ESP serial status text
    with shared_lock:
        selected_mode = selected_camera_mode
        connected = esp_connected
        active_mode = esp_active_mode
        last_message = esp_last_message

    if connected:
        esp_text = f"Selected={selected_mode} | ESP: Connected | Active={active_mode} | {last_message}"
        esp_color = (0, 255, 0) if active_mode == MODE_WEBCAM else (0, 255, 255)

    else:
        esp_text = f"Selected={selected_mode} | ESP: Disconnected | {last_message}"
        esp_color = (0, 0, 255)

    draw_transparent_text(
        frame=frame,
        text=esp_text,
        org=(20, frame.shape[0] - 15),
        font_scale=0.42,
        text_color=esp_color,
        thickness=1,
        bg_color=(0, 0, 0),
        alpha=0.30,
        padding=5
    )


# ==========================================================
# Blue Baseline Capture / Calibration
# ==========================================================

def start_blue_baseline_capture():
    global state, frozen_frame, current_live_frame

    if current_live_frame is None:
        print("[Warning] No live frame available yet.")
        return

    frozen_frame = current_live_frame.copy()
    state = STATE_PICK_COLOR

    print("Image frozen.")
    print("Click one point on the actual BLUE target color to set the baseline.")


def calibrate_blue_from_click(frame, x, y):
    global color_calibrated, sampled_hsv, HSV_LOWER, HSV_UPPER

    h_img, w_img = frame.shape[:2]

    x1 = max(0, x - COLOR_SAMPLE_RADIUS)
    x2 = min(w_img, x + COLOR_SAMPLE_RADIUS + 1)
    y1 = max(0, y - COLOR_SAMPLE_RADIUS)
    y2 = min(h_img, y + COLOR_SAMPLE_RADIUS + 1)

    patch = frame[y1:y2, x1:x2]

    if patch.size == 0:
        print("[Warning] Invalid color sample area.")
        return False

    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    pixels = hsv_patch.reshape(-1, 3)

    # Choose the most saturated and brightest pixels from the patch.
    # This helps when the click area contains both blue light and dark background.
    score = pixels[:, 1].astype(np.int32) + pixels[:, 2].astype(np.int32)
    top_count = max(5, len(pixels) // 3)
    top_pixels = pixels[np.argsort(score)[-top_count:]]

    median_hsv = np.median(top_pixels, axis=0).astype(int)

    h, s, v = int(median_hsv[0]), int(median_hsv[1]), int(median_hsv[2])

    h_low = max(0, h - H_TOL)
    h_high = min(179, h + H_TOL)

    s_low = max(MIN_S_LOWER, s - S_TOL)
    v_low = max(MIN_V_LOWER, v - V_TOL)

    HSV_LOWER = np.array([h_low, s_low, v_low])
    HSV_UPPER = np.array([h_high, 255, 255])

    sampled_hsv = (h, s, v)
    color_calibrated = True

    print("Blue color baseline calibrated.")
    print(f"Sampled HSV: H={h}, S={s}, V={v}")
    print(f"HSV lower: {HSV_LOWER.tolist()}")
    print(f"HSV upper: {HSV_UPPER.tolist()}")

    if s < 50 or v < 80:
        print("[Warning] Selected point may not be bright/saturated enough. Try clicking the center of the blue light.")

    return True


# ==========================================================
# ROI Mask Helper
# ==========================================================

def rebuild_roi_masks(frame_shape):
    global roi_masks, roi_masks_shape, roi_masks_dirty

    h, w = frame_shape[:2]

    roi_masks = []

    for pt in target_points:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, pt, ROI_RADIUS, 255, -1)
        roi_masks.append(mask)

    roi_masks_shape = (h, w)
    roi_masks_dirty = False


# ==========================================================
# Mouse Callback
# ==========================================================

def mouse_callback(event, x, y, flags, param):
    global target_points
    global state, frozen_frame
    global last_detected_pos
    global roi_masks_dirty
    global shared_detected_pos

    if event == cv2.EVENT_LBUTTONDOWN:

        # While choosing blue baseline from frozen image
        if state == STATE_PICK_COLOR:
            if frozen_frame is None:
                print("[Warning] No frozen frame available.")
                return

            success = calibrate_blue_from_click(frozen_frame, x, y)

            if success:
                state = STATE_DETECT
                frozen_frame = None
                print("Live image resumed.")
                print("Now click target positions 1-6.")

            return

        # Capture button on live image
        if state == STATE_WAIT_COLOR and is_capture_button_clicked(x, y):
            start_blue_baseline_capture()
            return

        # Before blue baseline is calibrated, do not allow target point setting
        if state == STATE_WAIT_COLOR:
            print("Please capture and select the blue baseline first.")
            return

        # After calibration, click target points 1-6
        if state == STATE_DETECT:
            if len(target_points) < MAX_POINTS:
                target_points.append((x, y))
                roi_masks_dirty = True
                print(f"Set Position {len(target_points)} at: {(x, y)}")
            else:
                print("Maximum 6 points already set. Press 'r' to reset.")

    elif event == cv2.EVENT_RBUTTONDOWN:

        if state == STATE_PICK_COLOR:
            frozen_frame = None
            state = STATE_DETECT if color_calibrated else STATE_WAIT_COLOR
            print("Blue baseline selection cancelled.")
            return

        target_points = []
        roi_masks_dirty = True
        last_detected_pos = None

        with shared_lock:
            shared_detected_pos = None

        print("Target points reset by right-click.")


# ==========================================================
# Esp Serial Communication
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
                    error_text = (
                        f"Port {port_name} busy/access denied. Close Arduino Serial Monitor or other serial apps."
                    )

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

def detect_calibrated_target_position(frame):
    global HSV_LOWER, HSV_UPPER
    global roi_masks_dirty, roi_masks_shape

    if HSV_LOWER is None or HSV_UPPER is None:
        return None, np.zeros(frame.shape[:2], dtype=np.uint8), []

    if len(target_points) == 0:
        return None, np.zeros(frame.shape[:2], dtype=np.uint8), []

    h, w = frame.shape[:2]

    if roi_masks_dirty or roi_masks_shape != (h, w):
        rebuild_roi_masks(frame.shape)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

    # Slightly expand detected color area
    color_mask = cv2.dilate(color_mask, kernel, iterations=1)

    best_pos = None
    best_score = 0
    scores = []

    for i, mask in enumerate(roi_masks):
        roi_color = cv2.bitwise_and(color_mask, color_mask, mask=mask)
        color_pixels = cv2.countNonZero(roi_color)

        max_area = 0

        # Only find contours when pixel count already passes threshold.
        # This saves CPU.
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
                best_pos = i + 1

    return best_pos, color_mask, scores


# ==========================================================
# Main Program
# ==========================================================

camera = CameraStream(CAMERA_INDEX)

if not camera.start():
    exit()

serial_thread = threading.Thread(target=serial_worker, daemon=True)
serial_thread.start()

window_name = "Color Detection and ESP32 Communication"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

print("Program started.")
print("YOLO removed.")
print("Optimized for smoother camera display.")
print("Workflow:")
print("1. Live image starts first.")
print("2. Press 'c' or click the Blue Baseline button.")
print("3. Image freezes.")
print("4. Click one point on the BLUE target color.")
print("5. Live image resumes.")
print("6. Click target positions 1-6.")
print("Controls:")
print("- c = capture still image for blue baseline")
print("- r = reset target points")
print("- m = show/hide calibrated color mask")
print("- w = Webcam mode (PC serial to ESP32)")
print("- h = Husky mode (ESP32 fallback camera)")
print("- q = quit")
print("- right-click = reset target points, or cancel baseline selection if image is frozen")
print(f"ESP serial protocol: MODE,<WEB|HUSKY> and DATA,<target 0-6>,<state 0-1> @ {SERIAL_BANDWIDTH}.")

try:
    while True:

        # ======================================================
        # 1. Select live frame or frozen frame
        # ======================================================

        if state == STATE_PICK_COLOR and frozen_frame is not None:
            frame = frozen_frame.copy()
        else:
            live_frame = camera.read()

            if live_frame is None:
                time.sleep(0.005)
                continue

            current_live_frame = live_frame.copy()
            frame = live_frame.copy()

        # ======================================================
        # 2. Detect calibrated color target
        # ======================================================

        detected_pos = None
        color_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if state == STATE_DETECT and color_calibrated:
            detected_pos, color_mask, _ = detect_calibrated_target_position(frame)

        with shared_lock:
            shared_detected_pos = detected_pos

        # ======================================================
        # 3. Draw target points and detection zones
        # ======================================================

        for i, pt in enumerate(target_points):
            x, y = pt

            color = (0, 255, 0)
            label = f"P{i + 1}"

            if detected_pos == i + 1:
                color = (255, 0, 0)

            cv2.circle(frame, pt, 5, color, -1)
            cv2.circle(frame, pt, ROI_RADIUS, color, 2)

            cv2.putText(
                frame,
                label,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA
            )

        # Console-only detection notice
        if state == STATE_DETECT and detected_pos is not None:
            if detected_pos != last_detected_pos:
                print(f"Calibrated blue target detected at Position: {detected_pos}")
                last_detected_pos = detected_pos
        else:
            last_detected_pos = None

        # ======================================================
        # 4. Draw Minimal UI Only
        # ======================================================

        draw_minimal_status(frame)

        if state == STATE_WAIT_COLOR:
            draw_capture_button(frame)

        cv2.imshow(window_name, frame)

        if SHOW_MASK:
            cv2.imshow("Calibrated Color Mask", color_mask)
        else:
            try:
                cv2.destroyWindow("Calibrated Color Mask")
            except cv2.error:
                pass

        # ======================================================
        # 5. Keyboard Controls
        # ======================================================

        key_code = cv2.waitKeyEx(1)
        key = normalize_key(key_code)

        if key == "q":
            break

        elif key == "c":
            if state != STATE_PICK_COLOR:
                start_blue_baseline_capture()

        elif key == "r":
            target_points = []
            roi_masks_dirty = True
            last_detected_pos = None

            with shared_lock:
                shared_detected_pos = None

            print("Target points reset. Please click points again.")

        elif key == "m":
            SHOW_MASK = not SHOW_MASK
            print(f"SHOW_MASK = {SHOW_MASK}")

        elif key == "w":
            set_camera_mode(MODE_WEBCAM)

        elif key == "h":
            set_camera_mode(MODE_HUSKY)

        elif key_code == 27:
            if state == STATE_PICK_COLOR:
                frozen_frame = None
                state = STATE_DETECT if color_calibrated else STATE_WAIT_COLOR
                print("Blue baseline selection cancelled.")

finally:
    stop_event.set()
    camera.stop()
    serial_thread.join(timeout=1.0)
    cv2.destroyAllWindows()
