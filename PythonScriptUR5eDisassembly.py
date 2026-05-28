"""
Battery Module Disassembly System
- Configure camera ROI and grid via OpenCV GUI
- Start disassembly with a button
- Robot removes lid, camera takes picture, vision classifies modules
- Robot removes ONLY damaged modules
- Modules numbered 1-8
"""

import cv2
import numpy as np
import json
import os
import threading
import time

from robodk.robolink import Robolink, ITEM_TYPE_ROBOT, ITEM_TYPE_TARGET
from robodk.robomath import transl

# ===========================================================================
# CONFIG
# ===========================================================================

CONFIG_FILE = "scanner_config.json"
DEFAULT_CONFIG = {
    "ROI_X": 271, "ROI_Y": 226,
    "ROI_W": 150, "ROI_H": 120,
    "GRID_COLS": 4, "GRID_ROWS": 2,
    "PAD_L": 4,  "PAD_R": 4,
    "PAD_T": 6,  "PAD_B": 6,
    "CELL_GAP": 4,
    "SAMPLE_MARGIN": 0.10,
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)
            merged = DEFAULT_CONFIG.copy()
            merged.update(data)
            print(f"[Config] Loaded from {CONFIG_FILE}")
            return merged
        except Exception as e:
            print(f"[Config] Load failed ({e}), using defaults.")
    return DEFAULT_CONFIG.copy()

def save_config(c):
    out = {k: (int(v) if k != "SAMPLE_MARGIN" else round(float(v), 4))
           for k, v in c.items()}
    with open(CONFIG_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print("[Config] Saved.")

def roi_ints(c):
    return int(c["ROI_X"]), int(c["ROI_Y"]), int(c["ROI_W"]), int(c["ROI_H"])

def grid_ints(c):
    return max(1, int(c["GRID_COLS"])), max(1, int(c["GRID_ROWS"]))

def pad_ints(c):
    return int(c["PAD_L"]), int(c["PAD_R"]), int(c["PAD_T"]), int(c["PAD_B"])

# ===========================================================================
# ROBOT CONFIG
# ===========================================================================

ROBOT_NAME        = "UR5e"
HOME_TARGET       = "Home"
X_OFFSET          = 47
Y_OFFSET          = 71.5
MODULE_TARGET     = "Module 1"
MODULE_APPROACH   = "App. md 1"
APPROACH_TARGET   = "Approach"
LID_TARGET        = "Lid"
LID_DISC_APPROACH = "App disc. lid"
LID_DISCARD       = "Discard lid"
SORT_APPROACH     = "App r. 1"
GOOD              = "Good 1"
BAD               = "Bad 1"

# Sort area grid layout: 4 cols x 2 rows, same offsets as pick tray
# Sequential placement order: position 0,1,2,... mapped to (col, row)
SORT_PATH = [
    (0, 0),  # Sort pos 1
    (1, 0),  # Sort pos 2
    (2, 0),  # Sort pos 3
    (3, 0),  # Sort pos 4
    (0, 1),  # Sort pos 5
    (1, 1),  # Sort pos 6
    (2, 1),  # Sort pos 7
    (3, 1),  # Sort pos 8
]

GRID_PATH = [
    (0, 0),  # Slot 1  — bottom row reversed: rightmost col = slot 1
    (1, 0),  # Slot 2
    (2, 0),  # Slot 3
    (3, 0),  # Slot 4  — leftmost col = slot 4
    (0, 1),  # Slot 5  — top row reversed: rightmost col = slot 5
    (1, 1),  # Slot 6
    (2, 1),  # Slot 7
    (3, 1),  # Slot 8  — leftmost col = slot 8
]

# ===========================================================================
# COLOUR THRESHOLDS
# ===========================================================================

blue_lower = np.array([95,  130,  30])
blue_upper = np.array([130, 255, 200])
red_lower1 = np.array([0,   120,  50])
red_upper1 = np.array([10,  255, 200])
red_lower2 = np.array([160, 120,  50])
red_upper2 = np.array([180, 255, 200])
BLUE_THRESH = 0.08
RED_THRESH  = 0.08

# ===========================================================================
# VISION HELPERS
# ===========================================================================

def grid_geometry(c):
    rx, ry, rw, rh = roi_ints(c)
    pl, pr, pt, pb = pad_ints(c)
    cols, rows = grid_ints(c)
    gap = int(c["CELL_GAP"])
    gx, gy = rx + pl, ry + pt
    gw, gh = max(1, rw - pl - pr), max(1, rh - pt - pb)
    cw, ch = gw / cols, gh / rows
    cells = []
    for row in range(rows):
        for col in range(cols):
            x1 = int(gx + col * cw) + gap
            y1 = int(gy + row * ch) + gap
            x2 = int(gx + (col + 1) * cw) - gap
            y2 = int(gy + (row + 1) * ch) - gap
            cells.append((row, col, x1, y1, x2, y2))
    return cells

def classify_cell(hsv, x1, y1, x2, y2, margin):
    mw = int((x2 - x1) * margin)
    mh = int((y2 - y1) * margin)
    cell = hsv[y1 + mh: y2 - mh, x1 + mw: x2 - mw]
    total = cell.shape[0] * cell.shape[1]
    if total == 0:
        return None

    blue_px = cv2.countNonZero(cv2.inRange(cell, blue_lower, blue_upper))
    red_px  = (cv2.countNonZero(cv2.inRange(cell, red_lower1, red_upper1)) +
               cv2.countNonZero(cv2.inRange(cell, red_lower2, red_upper2)))

    br, rr = blue_px / total, red_px / total

    if rr >= RED_THRESH and rr >= br:
        return True   # DAMAGED
    if br >= BLUE_THRESH:
        return False  # HEALTHY
    return None

def scan_frame(frame, c):
    """Capture + classify all slots. Returns (annotated_frame, mods_dict).
    mods keys are 1-based slot numbers (1–8).
    """
    result = frame.copy()
    hsv    = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    cols, rows = grid_ints(c)
    cells  = grid_geometry(c)
    margin = float(c["SAMPLE_MARGIN"])
    mods   = {}

    rx, ry, rw, rh = roi_ints(c)
    cv2.rectangle(result, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

    for (row, col, x1, y1, x2, y2) in cells:
        # Both rows reversed: rightmost col = lowest slot number in each row
        effective_col = (cols - 1 - col)
        old_index = effective_col + (rows - 1 - row) * cols
        slot = old_index + 1

        state = classify_cell(hsv, x1, y1, x2, y2, margin)
        mods[slot] = state

        if state is True:
            colour, label = (0, 0, 255), "DMG"
        elif state is False:
            colour, label = (255, 0, 0), "OK"
        else:
            colour, label = (100, 100, 100), ""

        cv2.rectangle(result, (x1, y1), (x2, y2), colour, 2)
        if label:
            cv2.putText(result, label, (x1 + 3, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1)
        cv2.putText(result, str(slot), (x2 - 14, y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1)

    healthy = sum(1 for v in mods.values() if v is False)
    damaged = sum(1 for v in mods.values() if v is True)

    cv2.putText(result, f"Healthy: {healthy}  |  Damaged: {damaged}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2)
    cv2.putText(result, "SCAN COMPLETE", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return result, mods

def draw_grid_preview(frame, c):
    rx, ry, rw, rh = roi_ints(c)
    cells = grid_geometry(c)
    cols, rows = grid_ints(c)

    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
    cv2.putText(frame, "SCAN ZONE", (rx, ry - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    for (row, col, x1, y1, x2, y2) in cells:
        # Both rows reversed: rightmost col = lowest slot number in each row
        effective_col = (cols - 1 - col)
        old_index = effective_col + (rows - 1 - row) * cols
        slot = old_index + 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), 2)
        cv2.putText(frame, str(slot), (x2 - 14, y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1)

# ===========================================================================
# ROBOT FUNCTIONS
# ===========================================================================

def open_gripper(robot):
    robot.setDO(0, 0)
    robot.setDO(1, 1)
    time.sleep(0.15)

def close_gripper(robot):
    robot.setDO(1, 0)
    robot.setDO(0, 1)
    time.sleep(0.15)

def fetch_targets(RDK):
    target_names = [
        HOME_TARGET, MODULE_TARGET, MODULE_APPROACH,
        APPROACH_TARGET, LID_TARGET, LID_DISC_APPROACH, LID_DISCARD,
        SORT_APPROACH,
    ]
    targets = {}
    for name in target_names:
        item = RDK.Item(name, ITEM_TYPE_TARGET)
        if not item.Valid():
            raise Exception(f"Target not found: '{name}'")
        targets[name] = item
        print(f"✓ Fetched target: {name}")
    return targets

def lid_removal(robot, targets):
    home_pose      = targets[HOME_TARGET].Pose()
    approach_pose  = targets[APPROACH_TARGET].Pose()
    lid_pose       = targets[LID_TARGET].Pose()
    disc_app_pose  = targets[LID_DISC_APPROACH].Pose()
    discard_pose   = targets[LID_DISCARD].Pose()
    lift_away_pose = discard_pose * transl(0, 0, -60)

    open_gripper(robot)
    robot.MoveJ(home_pose)
    robot.MoveL(lid_pose)
    close_gripper(robot)
    robot.MoveL(approach_pose)
    robot.MoveL(disc_app_pose)
    robot.MoveL(discard_pose)
    open_gripper(robot)
    robot.MoveL(lift_away_pose)

def module_sort(robot, targets, mods):
    """Pick ONLY damaged slots (state is True) and place them sequentially
    into sort positions 1, 2, 3... regardless of which pick slot they came from.
    GRID_PATH defines the pick traversal order (slot 1→8).
    SORT_PATH defines the place positions used in order.
    """
    approach_pose  = targets[MODULE_APPROACH].Pose()
    sort_base_pose = targets[SORT_APPROACH].Pose()

    sort_step = 0  # increments only when a module is actually placed

    for step, (pick_col, pick_row) in enumerate(GRID_PATH):
        slot = step + 1

        state = mods.get(slot)

        if state is not True:
            if state is False:
                print(f"  Slot {slot}: healthy, skipping.")
            else:
                print(f"  Slot {slot}: empty, skipping.")
            continue

        # Pick pose — offset from MODULE_APPROACH by tray grid position
        hover = approach_pose * transl(-pick_col * X_OFFSET, -pick_row * Y_OFFSET, 0)
        down  = hover         * transl(0, 1, 68)

        # Place pose — next sequential sort position
        sort_col, sort_row = SORT_PATH[sort_step]
        sort_hover = sort_base_pose * transl(-sort_col * X_OFFSET, -sort_row * Y_OFFSET, 0)
        sort_down  = sort_hover     * transl(0, 0, 68)

        # Pick
        robot.setRounding(10)
        robot.MoveL(hover)
        robot.setRounding(0)
        robot.MoveL(down)
        close_gripper(robot)
        robot.MoveL(hover)

        # Place
        robot.MoveL(sort_hover)
        robot.setRounding(0)
        robot.MoveL(sort_down)
        open_gripper(robot)
        robot.MoveL(sort_hover)

        print(f"  Slot {slot}: DAMAGED → placed at sort position {sort_step + 1}.")
        sort_step += 1

# ===========================================================================
# CONTROL STRIP  (drawn below the camera image)
# ===========================================================================

BUTTON_H   = 40
BUTTON_GAP = 10

def make_control_strip(width, status_text, phase, cfg_open):
    strip = np.zeros((BUTTON_H * 2 + BUTTON_GAP * 3, width, 3), dtype=np.uint8)
    strip[:] = (30, 30, 30)

    cv2.putText(strip, status_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)

    buttons = []

    def add_btn(label, color, enabled=True):
        x = BUTTON_GAP + len(buttons) * (130 + BUTTON_GAP)
        y = BUTTON_H + BUTTON_GAP
        w, h = 130, BUTTON_H - 4
        c = color if enabled else (60, 60, 60)

        cv2.rectangle(strip, (x, y), (x + w, y + h), c, -1)
        cv2.rectangle(strip, (x, y), (x + w, y + h), (200, 200, 200), 1)

        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        tx = x + (w - tw) // 2

        cv2.putText(strip, label, (tx, y + h - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (255, 255, 255) if enabled else (100, 100, 100), 1)

        buttons.append({"label": label, "x": x, "w": w,
                        "strip_y": y, "h": h})

    if phase in ("config", "waiting", "scanned"):
        add_btn("Config [C]",  (0, 130, 130), not cfg_open)
        add_btn("Save [S]",    (30, 100, 30))
        add_btn("START [SPC]", (0, 160, 40))
        add_btn("Quit [Q]",    (100, 30, 30))
    elif phase == "running":
        add_btn("Config [C]",  (60, 60, 60), False)
        add_btn("Save [S]",    (60, 60, 60), False)
        add_btn("START [SPC]", (60, 60, 60), False)
        add_btn("Quit [Q]",    (100, 30, 30))

    if phase == "scanned":
        buttons.clear()
        strip[BUTTON_H + BUTTON_GAP:, :] = (30, 30, 30)
        add_btn("Retake [R]",  (0, 130, 200))
        add_btn("Config [C]",  (0, 130, 130), not cfg_open)
        add_btn("Save [S]",    (30, 100, 30))
        add_btn("Quit [Q]",    (100, 30, 30))

    return strip, buttons

# ===========================================================================
# MAIN APPLICATION
# ===========================================================================

def main():
    # ── Camera ──────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, _frame = cap.read()
    CAM_H, CAM_W = (_frame.shape[:2] if ret else (480, 640))

    cfg = load_config()

    # ── RoboDK ──────────────────────────────────────────────────────────────
    RDK        = Robolink()
    robot_item = RDK.Item(ROBOT_NAME, ITEM_TYPE_ROBOT)

    if not robot_item.Valid():
        raise Exception(f"Robot '{ROBOT_NAME}' not found in RoboDK.")

    robot = robot_item
    robot.setSpeed(speed_linear=1600, speed_joints=1200,
                   accel_linear=4600, accel_joints=550)
    targets = fetch_targets(RDK)

    # ── App state ────────────────────────────────────────────────────────────
    phase         = "config"
    status_text   = "READY — Configure scan zone, then press START"
    scanned_frame = None
    mods          = {}
    cfg_open      = False
    _quit         = [False]

    last_buttons  = [[]]

    # ── Windows ──────────────────────────────────────────────────────────────
    MAIN_WIN = "Battery Disassembly System"
    CFG_WIN  = "Configuration"
    cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL)

    # ── Drag state ───────────────────────────────────────────────────────────
    drag_start = [None]
    drag_end   = [None]
    dragging   = [False]

    # ── Button handler ───────────────────────────────────────────────────────
    def handle_button(label):
        nonlocal phase, status_text, cfg_open, scanned_frame, mods
        label = label.strip()

        if "Config" in label:
            if phase == "running":
                return
            if cfg_open:
                close_config_window()
            else:
                open_config_window()

        elif "Save" in label:
            save_config(cfg)
            status_text = "Config saved ✓"

        elif "START" in label or "SPC" in label:
            if phase not in ("config", "waiting", "scanned"):
                return
            phase = "running"
            status_text = "Running disassembly…"
            t = threading.Thread(target=run_disassembly, daemon=True)
            t.start()

        elif "Retake" in label:
            phase = "waiting"
            status_text = "Ready — press START to begin a new batch"
            scanned_frame = None
            mods.clear()

        elif "Quit" in label:
            _quit[0] = True

    # ── Mouse callback ───────────────────────────────────────────────────────
    def click_event(event, x, y, flags, param):
        strip_y = CAM_H

        if y >= strip_y:
            if event == cv2.EVENT_LBUTTONDOWN:
                rel_y = y - strip_y
                for btn in last_buttons[0]:
                    by = btn["strip_y"]
                    if (btn["x"] <= x <= btn["x"] + btn["w"] and
                            by <= rel_y <= by + btn["h"]):
                        handle_button(btn["label"])
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start[0] = (x, y)
            drag_end[0] = (x, y)
            dragging[0] = True

        elif event == cv2.EVENT_MOUSEMOVE and dragging[0]:
            drag_end[0] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and dragging[0]:
            drag_end[0] = (x, y)
            dragging[0] = False
            x1, y1 = drag_start[0]
            x2, y2 = drag_end[0]

            if abs(x2 - x1) > 8 and abs(y2 - y1) > 8:
                cfg["ROI_X"] = min(x1, x2)
                cfg["ROI_Y"] = min(y1, y2)
                cfg["ROI_W"] = abs(x2 - x1)
                cfg["ROI_H"] = abs(y2 - y1)

                if cfg_open:
                    cv2.setTrackbarPos("ROI  X",      CFG_WIN, cfg["ROI_X"])
                    cv2.setTrackbarPos("ROI  Y",      CFG_WIN, cfg["ROI_Y"])
                    cv2.setTrackbarPos("ROI  Width",  CFG_WIN, cfg["ROI_W"])
                    cv2.setTrackbarPos("ROI  Height", CFG_WIN, cfg["ROI_H"])

            drag_start[0] = drag_end[0] = None

    cv2.setMouseCallback(MAIN_WIN, click_event)

    # ── Config window ────────────────────────────────────────────────────────
    def open_config_window():
        nonlocal cfg_open
        cv2.namedWindow(CFG_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CFG_WIN, 380, 780)

        def tb(label, key, max_val, scale=1):
            init = int(round(float(cfg[key]) * scale))
            def cb(val):
                cfg[key] = val / scale
            cv2.createTrackbar(label, CFG_WIN, init, max_val, cb)

        tb("ROI  X",           "ROI_X",         CAM_W)
        tb("ROI  Y",           "ROI_Y",         CAM_H)
        tb("ROI  Width",       "ROI_W",         CAM_W)
        tb("ROI  Height",      "ROI_H",         CAM_H)
        tb("Grid  Columns",    "GRID_COLS",     16)
        tb("Grid  Rows",       "GRID_ROWS",     10)
        tb("Padding  Left",    "PAD_L",         80)
        tb("Padding  Right",   "PAD_R",         80)
        tb("Padding  Top",     "PAD_T",         80)
        tb("Padding  Bottom",  "PAD_B",         80)
        tb("Cell  Gap",        "CELL_GAP",      30)
        tb("Sample  Margin %", "SAMPLE_MARGIN", 49, scale=100)
        cfg_open = True

    def close_config_window():
        nonlocal cfg_open
        cv2.destroyWindow(CFG_WIN)
        cfg_open = False

    def make_config_panel():
        panel = np.zeros((280, 380, 3), dtype=np.uint8)
        panel[:] = (28, 28, 28)

        cols, rows = grid_ints(cfg)
        rx, ry, rw, rh = roi_ints(cfg)
        pl, pr, pt, pb = pad_ints(cfg)

        lines = [
            ("LIVE VALUES",                                        (0, 220, 220)),
            ("",                                                   None),
            (f"Scan zone   x={rx}  y={ry}  {rw} x {rh} px",      (180, 230, 180)),
            (f"Grid        {cols} cols  x  {rows} rows",          (180, 230, 180)),
            (f"Padding     L={pl}  R={pr}  T={pt}  B={pb}",       (180, 230, 180)),
            (f"Cell gap    {int(cfg['CELL_GAP'])} px",            (180, 230, 180)),
            (f"Sample margin  {float(cfg['SAMPLE_MARGIN']):.0%}", (180, 230, 180)),
            ("",                                                   None),
            ("Drag on camera window to reposition ROI.",          (140, 140, 140)),
            ("Press S to save config.",                           (140, 140, 140)),
        ]

        y = 22
        for text, col in lines:
            if col:
                cv2.putText(panel, text, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)
            y += 24
        return panel

    # ── Disassembly worker ───────────────────────────────────────────────────
    def run_disassembly():
        from robodk.robolink import RUNMODE_RUN_ROBOT
        RDK.setRunMode(RUNMODE_RUN_ROBOT)  # ← re-set every time
        robot.Connect()

        robot.setSpeed(speed_linear=1600, speed_joints=1200,
                   accel_linear=4600, accel_joints=550)
        nonlocal phase, status_text, scanned_frame, mods

        try:
            status_text = "Step 1/3 — Removing lid…"
            print("\n=== Step 1: Lid Removal ===")
            lid_removal(robot, targets)

            status_text = "Step 2/3 — Scanning modules…"
            print("\n=== Step 2: Camera Scan ===")
            time.sleep(0.5)

            ret, frame = cap.read()
            if not ret:
                status_text = "ERROR: Camera read failed"
                phase = "waiting"
                return

            result, scan_mods = scan_frame(frame, cfg)
            scanned_frame = result
            mods.clear()
            mods.update(scan_mods)

            healthy = sum(1 for v in mods.values() if v is False)
            damaged = sum(1 for v in mods.values() if v is True)

            for slot in sorted(mods.keys()):
                state = mods[slot]
                s = "DAMAGED" if state is True else ("HEALTHY" if state is False else "empty")
                print(f"  Slot {slot}: {s}")
            print(f"  → {healthy} healthy, {damaged} damaged")

            status_text = f"Step 3/3 — Removing {damaged} damaged module(s)…"
            print("\n=== Step 3: Removing Damaged Modules ===")
            robot.MoveL(targets[MODULE_APPROACH].Pose())
            module_sort(robot, targets, mods)
            robot.MoveJ(targets[HOME_TARGET].Pose())

            status_text = (f"Done ✓   {damaged} damaged removed  |  "
                           f"{healthy} healthy left in tray   — press START for next batch")
            phase = "scanned"
            print("\n=== Disassembly Complete ===")

        except Exception as e:
            status_text = f"ERROR: {e}"
            print(f"[ERROR] {e}")
            phase = "waiting"

    # ── Main loop ────────────────────────────────────────────────────────────
    print("=== Battery Disassembly System ===")
    print("  Configure ROI, then SPACE / click START to begin.")
    print("  Modules numbered 1–8  |  Only DAMAGED modules will be removed.")

    while not _quit[0]:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c') and phase != "running":
            if cfg_open:
                close_config_window()
            else:
                open_config_window()
        elif key == ord('s'):
            save_config(cfg)
            status_text = "Config saved ✓"
        elif key == ord(' ') and phase in ("config", "waiting", "scanned"):
            handle_button("START")
        elif key == ord('r') and phase == "scanned":
            handle_button("Retake")

        if phase in ("running", "scanned") and scanned_frame is not None:
            cam_display = scanned_frame.copy()
        else:
            ret, raw = cap.read()
            cam_display = raw.copy() if ret else np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
            draw_grid_preview(cam_display, cfg)

            if dragging[0] and drag_start[0] and drag_end[0]:
                x1, y1 = drag_start[0]
                x2, y2 = drag_end[0]
                cv2.rectangle(cam_display,
                              (min(x1, x2), min(y1, y2)),
                              (max(x1, x2), max(y1, y2)),
                              (0, 255, 0), 2)

        badge_col = {
            "config":  (0, 200, 200),
            "waiting": (0, 200, 0),
            "running": (0, 140, 255),
            "scanned": (255, 200, 0)
        }.get(phase, (200, 200, 200))

        cv2.putText(cam_display, f"Phase: {phase.upper()}",
                    (cam_display.shape[1] - 200, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_col, 1)

        strip, buttons = make_control_strip(cam_display.shape[1],
                                            status_text, phase, cfg_open)
        last_buttons[0] = buttons

        cv2.imshow(MAIN_WIN, np.vstack([cam_display, strip]))

        if cfg_open:
            cv2.imshow(CFG_WIN, make_config_panel())

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
