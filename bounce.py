import cv2, numpy as np, random, time, os, json
from collections import deque
 
# ============================== 1. CONFIG ==============================
W, H = 640, 480
R = 12                                                    # ball radius
GRAVITY, BOUNCE, MAXV = 700, 0.65, 650                    # gravity, bounciness, speed limit (px/s)
K5 = np.ones((5, 5), np.uint8)
yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
DISK = xx ** 2 + yy ** 2 <= R * R                         # ball footprint used for collision checks
FONT = cv2.FONT_HERSHEY_SIMPLEX
GAME_WIN, MASK_WIN, CTRL_WIN = "BALL GUIDE", "Mask", "Controls"

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
HSV_FILE = os.path.join(HERE, "hsv_config.json")

# --- HSV settings (edited live from the "Controls" window) ---
HSV_KEYS = [("H min", 179), ("H max", 179), ("S min", 255), ("S max", 255), ("V min", 255), ("V max", 255)]
HSV = {"H min": 10, "H max": 35, "S min": 60, "S max": 255, "V min": 80, "V max": 255}   # default = yellow

def load_hsv():
    try:
        with open(HSV_FILE) as f:
            HSV.update({k: int(v) for k, v in json.load(f).items() if k in HSV})
    except Exception:
        pass

def save_hsv():
    try:
        with open(HSV_FILE, "w") as f:
            json.dump(HSV, f)
    except Exception:
        pass

def hsv_range():
    return (np.array([HSV["H min"], HSV["S min"], HSV["V min"]], np.uint8),
            np.array([HSV["H max"], HSV["S max"], HSV["V max"]], np.uint8))

def open_settings():
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(CTRL_WIN, W + 50, 300)
    for k, mx in HSV_KEYS:
        cv2.createTrackbar(k, CTRL_WIN, HSV[k], mx, lambda x: None)
    info = np.full((46, 460, 3), 30, np.uint8)
    cv2.putText(info, "H: 0-179   S, V: 0-255   |  white in 'Mask' = detected", (8, 28), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(CTRL_WIN, info)

def read_settings():
    for k, _ in HSV_KEYS:
        HSV[k] = cv2.getTrackbarPos(k, CTRL_WIN)

def close_settings():
    save_hsv()
    try:
        cv2.destroyWindow(CTRL_WIN)
    except cv2.error:
        pass

# ============================== 2. IMAGE PROCESSING & MASKING ==============================
def find_yellow(frame):
    """Camera frame -> (raw yellow mask, cleaned yellow mask)."""
    lo, hi = hsv_range()
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)             # 1. blur + convert to HSV
    raw = cv2.inRange(hsv, lo, hi)                                                        # 2. keep yellow pixels
    clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, K5, iterations=2)                       # 3. remove noise
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, K5, iterations=2)                    # 4. fill holes
    return raw, (clean if clean.sum() > 255 * 800 else np.zeros_like(clean))              # 5. too small = not the glove

def mask_view(frame, clean, final):
    """Jendela khusus untuk menampilkan hasil masking secara konsisten."""
    area = int(clean.sum() // 255)
    m = cv2.cvtColor(cv2.resize(clean, (320, 240)), cv2.COLOR_GRAY2BGR)
    seen = cv2.resize(cv2.bitwise_and(frame, frame, mask=clean), (320, 240))
    ok = final.any()
    cv2.putText(m, f"mask  area={area}px", (6, 20), FONT, 0.5, (0, 255, 0) if ok else (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(m, "DETECTED" if ok else "TOO SMALL (<800px)", (6, 40), FONT, 0.5, (0, 255, 0) if ok else (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(seen, "camera x mask", (6, 20), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack([m, seen])

# ============================== 3. GAME LOGIC ==============================
def make_level(score, target, tr):
    """Random bars + bumpers (as masks). Keeps the target area free."""
    while True:
        bars, bumps = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
        for _ in range(2 + min(score // 2, 3)):
            cx, cy = random.randint(80, W - 80), random.randint(150, H - 110)
            a, l = np.radians(random.uniform(-35, 35)), random.randint(45, 80)
            dx, dy = int(l * np.cos(a)), int(l * np.sin(a))
            cv2.line(bars, (cx - dx, cy - dy), (cx + dx, cy + dy), 255, 10)
        for _ in range(1 + min(score // 3, 2)):
            cv2.circle(bumps, (random.randint(80, W - 80), random.randint(150, H - 110)), 22, 255, -1)
        m = tr + R + 5
        if not (bars | bumps)[target[1] - m:target[1] + m, target[0] - m:target[0] + m].any():
            return bars > 0, bumps > 0
 
def collide(p, v, solid, vs, e=BOUNCE):
    """Push the ball out of `solid` (bool mask) and bounce it. vs = velocity of that object, e = bounciness."""
    x, y = int(p[0]), int(p[1])
    if x < R or y < R or x >= W - R or y >= H - R:
        return
    hit = solid[y - R:y + R + 1, x - R:x + R + 1] & DISK                                  # solid pixels touching the ball
    if hit.sum() < 4:
        return
    ys, xs = np.nonzero(hit)
    n = -np.array([xs.mean() - R, ys.mean() - R])                                         # normal: from object toward ball
    d = np.hypot(*n)
    n = n / d if d > 1e-3 else np.array([0.0, -1.0])
    rel = v - vs                                                                          # velocity relative to the object
    vn = rel @ n
    if vn < 0:                                                                            # moving into it -> bounce
        v[:] = vs + rel - (1 + e) * vn * n
    p += n * 2
 
def new_ball():
    return np.array([random.uniform(100, W - 100), 30.0]), np.zeros(2)
 
def new_target(tr):
    return (random.randint(80, W - 80), random.randint(160, H - 100)), tr
 
def label(img, s, y, size, color, thick=2):
    w = cv2.getTextSize(s, FONT, size, thick)[0][0]
    cv2.putText(img, s, ((W - w) // 2, y), FONT, size, color, thick, cv2.LINE_AA)

def draw_button(img, txt, c, r, prog):
    cv2.circle(img, c, r, (255, 230, 0), 3, cv2.LINE_AA)
    if prog > 0:
        cv2.ellipse(img, c, (r + 8, r + 8), -90, 0, int(360 * min(prog, 1.0)), (90, 255, 120), 5, cv2.LINE_AA)
    lines = txt.split("\n")
    sz = min(0.8, 1.5 * r / max(cv2.getTextSize(l, FONT, 1, 2)[0][0] for l in lines))
    lh = int(44 * sz)
    y0 = c[1] - (len(lines) - 1) * lh // 2 + int(8 * sz)
    for i, l in enumerate(lines):
        w = cv2.getTextSize(l, FONT, sz, 2)[0][0]
        cv2.putText(img, l, (c[0] - w // 2, y0 + i * lh), FONT, sz, (255, 255, 255), 2, cv2.LINE_AA)

_bm = {}
def overlap(mask, c, r):
    if (c, r) not in _bm:
        x0, y0, x1, y1 = max(c[0] - r, 0), max(c[1] - r, 0), min(c[0] + r + 1, W), min(c[1] + r + 1, H)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        _bm[(c, r)] = (x0, y0, x1, y1, (gx - c[0]) ** 2 + (gy - c[1]) ** 2 <= r * r)
    x0, y0, x1, y1, d = _bm[(c, r)]
    return int((mask[y0:y1, x0:x1] & d).sum())

# ============================== 4. MAIN LOOP ==============================
def main():
    load_hsv()
    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow(GAME_WIN)
    cv2.namedWindow(MASK_WIN)                                 # Jendela utama & Masking selalu terbuka
    cv2.moveWindow(GAME_WIN, 20, 20)
    cv2.moveWindow(MASK_WIN, W + 50, 20)
    
    state = "menu"                                            # menu -> play -> over
    p, v = new_ball()
    target, tr = new_target(34)
    bars_m = bump_m = np.zeros((H, W), bool)
    score, lives, still = 0, 3, 0.0
    hold = {}
    trail, last, pc, vp = deque(maxlen=14), time.time(), None, np.zeros(2)
    settings, s_age = False, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera not found")
            break
        frame = cv2.resize(cv2.flip(frame, 1), (W, H))
        dt, last = max(min(time.time() - last, 0.05), 1e-3), time.time()
    
        # --- HSV sliders -> detection settings ---
        if settings:
            s_age += 1
            closed = False
            try:
                if s_age > 5 and cv2.getWindowProperty(CTRL_WIN, cv2.WND_PROP_VISIBLE) < 1:
                    closed = True
                else:
                    read_settings()
            except cv2.error:
                closed = True
            if closed:
                settings = False
                close_settings()

        # --- image processing: find the glove + mask view ---
        raw, yellow = find_yellow(frame)
        cv2.imshow(MASK_WIN, mask_view(frame, raw, yellow))   # Window masking selalu ter-update
        
        solid_y = yellow > 0
        if solid_y.any():
            m = cv2.moments(yellow, True)
            c = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
            vp = np.clip(0.6 * vp + 0.4 * (c - pc) / dt, -900, 900) if pc is not None else np.zeros(2)
            pc = c                                            # velocity = distance moved between frames
        else:
            pc, vp = None, np.zeros(2)
    
        action = None
        btns = [("START", (215, 300), 55, "start"), ("SETTINGS", (425, 300), 55, "settings")] if state == "menu" else \
               [("RETRY", (215, 300), 55, "retry"), ("SETTINGS", (425, 300), 55, "settings")]

        if state == "play":
            # --- physics (4 sub-steps per frame) ---
            for _ in range(4):
                h = dt / 4
                v[1] += GRAVITY * h
                sp = np.hypot(*v)
                if sp > MAXV:
                    v *= MAXV / sp
                p += v * h
                if p[0] < R or p[0] > W - R:
                    v[0] *= -BOUNCE; p[0] = np.clip(p[0], R, W - R)
                if p[1] < R:
                    v[1] *= -BOUNCE; p[1] = R
                collide(p, v, bars_m, np.zeros(2))
                collide(p, v, bump_m, np.zeros(2), 1.25)      # bumpers add energy
                collide(p, v, solid_y, vp)                    # the glove paddle
            trail.append((int(p[0]), int(p[1])))
            still = still + dt if np.hypot(*v) < 25 else 0.0
            if np.hypot(*(p - target)) < tr + R:              # target hit -> next level
                score += 1
                target, tr = new_target(max(18, tr - 2))
                bars_m, bump_m = make_level(score, target, tr)
                p, v = new_ball(); trail.clear()
            elif p[1] > H + R or still > 3:                   # fell off the screen (or stuck)
                lives -= 0 if still > 3 else 1
                p, v = new_ball(); trail.clear(); still = 0.0
                state = "play" if lives > 0 else "over"
        else:
            # --- menu / game over: hold glove over button for 1 second ---
            best, bo = None, 0
            for _, c, r, a in btns:
                ov = overlap(solid_y, c, r)
                if ov > 300 and ov > bo:
                    best, bo = a, ov
            for _, _, _, a in btns:
                hold[a] = hold.get(a, 0.0) + dt if a == best else 0.0
            if best and hold[best] > 1.0:
                action = best
    
        # --- draw ---
        img = (frame * 0.35).astype(np.uint8)                 # dimmed camera as background
        neon = np.zeros_like(img)
        neon[bars_m] = (255, 230, 0)
        neon[bump_m] = (255, 80, 220)
        neon[solid_y] = (0, 220, 255)
        img = cv2.add(img, cv2.add(cv2.GaussianBlur(neon, (0, 0), 7), neon // 2))   # glow = blur + add
        
        if state == "play":
            for i, q in enumerate(trail):
                cv2.circle(img, q, 2 + i // 3, (255, 255, 255), -1, cv2.LINE_AA)
            r = tr + int(4 * np.sin(time.time() * 5))         # pulsing target
            cv2.circle(img, target, r, (90, 255, 120), 3, cv2.LINE_AA)
            cv2.circle(img, target, r // 3, (90, 255, 120), -1, cv2.LINE_AA)
            glow = np.zeros_like(img)
            cv2.circle(glow, (int(p[0]), int(p[1])), R + 6, (255, 255, 255), -1, cv2.LINE_AA)
            img = cv2.add(img, cv2.GaussianBlur(glow, (0, 0), 8))
            cv2.circle(img, (int(p[0]), int(p[1])), R, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(img, f"SCORE {score}   LIVES {lives}", (10, 28), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            label(img, "BALL GUIDE" if state == "menu" else f"GAME OVER  -  SCORE {score}", 110, 1.1, (255, 230, 0), 3)
            for txt, c, r, a in btns:
                draw_button(img, txt, c, r, hold.get(a, 0.0))
            label(img, "hold glove over button for 1s (or press SPACE / 1 / 2)  |  S = settings", H - 30, 0.5, (200, 200, 200), 1)
        
        cv2.imshow(GAME_WIN, img)
    
        # --- keyboard controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            action = "settings"
        elif key == ord('m'):
            state = "menu"
            hold.clear()
        elif key == ord('r'):
            action = "start" if state == "menu" else "retry"
        elif state != "play" and key in (32, ord('1'), ord('2')):
            idx = 0 if key in (32, ord('1')) else 1
            action = btns[idx][3]

        if action == "settings":
            if settings:
                settings = False
                close_settings()
            else:
                settings, s_age = True, 0
                open_settings()
            hold.clear()
        elif action in ("start", "retry"):
            p, v = new_ball(); trail.clear()
            score, lives, still = 0, 3, 0.0
            hold.clear()
            target, tr = new_target(34)
            bars_m, bump_m = make_level(0, target, tr)
            state = "play"

    save_hsv()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
