import cv2
import mediapipe as mp
import numpy as np
import time
import random

# INIT
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

captured = False
puzzle_ready = False

GRID = 3
TILE = 100

tiles = []
order = []
blank_idx = None

# =========================
# HELPER
# =========================
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def split_tiles(img):
    img = cv2.resize(img, (GRID*TILE, GRID*TILE))
    t = []
    for r in range(GRID):
        for c in range(GRID):
            t.append(img[r*TILE:(r+1)*TILE, c*TILE:(c+1)*TILE])
    return t

def shuffle_tiles():
    global order, blank_idx
    order = list(range(GRID*GRID))
    blank_idx = len(order) - 1

    for _ in range(100):
        neighbors = get_neighbors(blank_idx)
        swap = random.choice(neighbors)
        order[blank_idx], order[swap] = order[swap], order[blank_idx]
        blank_idx = swap

def get_neighbors(idx):
    r, c = divmod(idx, GRID)
    n = []
    if r > 0: n.append((r-1)*GRID + c)
    if r < GRID-1: n.append((r+1)*GRID + c)
    if c > 0: n.append(r*GRID + c-1)
    if c < GRID-1: n.append(r*GRID + c+1)
    return n

def draw_puzzle(frame):
    canvas = np.zeros((GRID*TILE, GRID*TILE, 3), dtype=np.uint8)
    for i, t_idx in enumerate(order):
        r, c = divmod(i, GRID)
        if t_idx == GRID*GRID-1:
            tile = np.zeros((TILE, TILE, 3), dtype=np.uint8)
        else:
            tile = tiles[t_idx]

        canvas[r*TILE:(r+1)*TILE, c*TILE:(c+1)*TILE] = tile

    cv2.imshow("PUZZLE", canvas)

# =========================
# MAIN LOOP
# =========================
start_capture_time = None

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    points = []

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # ambil titik jempol & telunjuk
            thumb = hand.landmark[4]
            index = hand.landmark[8]

            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(index.x * w), int(index.y * h)

            cv2.circle(frame, (tx, ty), 8, (0,255,255), -1)
            cv2.circle(frame, (ix, iy), 8, (0,255,255), -1)

            points.append((tx, ty))
            points.append((ix, iy))

    # =========================
    # BUAT KOTAK (2 tangan)
    # =========================
    if len(points) >= 4:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)

        # =========================
        # DETEKSI JENTIK (PINCH)
        # =========================
        if len(points) >= 4:
            d = distance(points[0], points[1])  # jempol-index

            if d < 40:
                if start_capture_time is None:
                    start_capture_time = time.time()

                # tahan 1 detik
                if time.time() - start_capture_time > 1 and not captured:
                    face = frame[y1:y2, x1:x2]

                    if face.size > 0:
                        tiles = split_tiles(face)
                        shuffle_tiles()
                        captured = True
                        puzzle_ready = True
                        print("Captured!")
            else:
                start_capture_time = None

    cv2.putText(frame, "Pinch (jempol + telunjuk) untuk capture", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("CAMERA", frame)

    if puzzle_ready:
        draw_puzzle(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
