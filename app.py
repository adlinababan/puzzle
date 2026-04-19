import cv2
import time
import random
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# =========================
# CONFIG
# =========================
GRID_SIZE = 3
TILE_SIZE = 120
BOARD_SIZE = GRID_SIZE * TILE_SIZE

st.set_page_config(page_title="Your Face Puzzle", layout="centered")

# =========================
# SESSION STATE
# =========================
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

if "tiles" not in st.session_state:
    st.session_state.tiles = None

if "blank_idx" not in st.session_state:
    st.session_state.blank_idx = None

if "solved_tiles" not in st.session_state:
    st.session_state.solved_tiles = None

if "started" not in st.session_state:
    st.session_state.started = False

if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "finished" not in st.session_state:
    st.session_state.finished = False

if "moves" not in st.session_state:
    st.session_state.moves = 0


# =========================
# HELPERS
# =========================
def crop_center_square(img: np.ndarray) -> np.ndarray:
    """Crop image to center square."""
    h, w = img.shape[:2]
    side = min(h, w)
    x = (w - side) // 2
    y = (h - side) // 2
    return img[y:y+side, x:x+side]


def prepare_face_image(img: np.ndarray) -> np.ndarray:
    """
    Resize image to board size.
    Input img expected in BGR from OpenCV.
    """
    img = crop_center_square(img)
    img = cv2.resize(img, (BOARD_SIZE, BOARD_SIZE))
    return img


def split_into_tiles(img: np.ndarray):
    """Split image into GRID_SIZE x GRID_SIZE tiles."""
    tiles = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            y1 = r * TILE_SIZE
            y2 = y1 + TILE_SIZE
            x1 = c * TILE_SIZE
            x2 = x1 + TILE_SIZE
            tile = img[y1:y2, x1:x2].copy()
            tiles.append(tile)
    return tiles


def get_solved_order():
    return list(range(GRID_SIZE * GRID_SIZE))


def get_neighbors(idx):
    r, c = divmod(idx, GRID_SIZE)
    neighbors = []
    if r > 0:
        neighbors.append((r - 1) * GRID_SIZE + c)
    if r < GRID_SIZE - 1:
        neighbors.append((r + 1) * GRID_SIZE + c)
    if c > 0:
        neighbors.append(r * GRID_SIZE + (c - 1))
    if c < GRID_SIZE - 1:
        neighbors.append(r * GRID_SIZE + (c + 1))
    return neighbors


def shuffle_safely(order, steps=150):
    """
    Shuffle by doing valid blank moves, so the puzzle is always solvable.
    Blank tile is last tile.
    """
    current = order.copy()
    blank = len(current) - 1

    for _ in range(steps):
        neighbors = get_neighbors(blank)
        swap_idx = random.choice(neighbors)
        current[blank], current[swap_idx] = current[swap_idx], current[blank]
        blank = swap_idx

    return current, blank


def create_blank_tile():
    tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    tile[:] = (20, 20, 20)
    cv2.rectangle(tile, (0, 0), (TILE_SIZE - 1, TILE_SIZE - 1), (120, 120, 120), 2)
    return tile


def render_board(tile_order, original_tiles):
    """
    Render board image based on current order.
    tile_order contains tile indexes, with last index treated as blank.
    """
    canvas = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)
    blank_tile_idx = GRID_SIZE * GRID_SIZE - 1

    for board_pos, tile_idx in enumerate(tile_order):
        r, c = divmod(board_pos, GRID_SIZE)
        y1 = r * TILE_SIZE
        y2 = y1 + TILE_SIZE
        x1 = c * TILE_SIZE
        x2 = x1 + TILE_SIZE

        if tile_idx == blank_tile_idx:
            tile = create_blank_tile()
        else:
            tile = original_tiles[tile_idx]

        # border
        tile_copy = tile.copy()
        cv2.rectangle(tile_copy, (0, 0), (TILE_SIZE - 1, TILE_SIZE - 1), (255, 255, 255), 1)

        canvas[y1:y2, x1:x2] = tile_copy

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def is_solved(tile_order):
    return tile_order == get_solved_order()


def move_tile(clicked_idx):
    """
    clicked_idx is board position.
    Swap if adjacent to blank.
    """
    blank = st.session_state.blank_idx
    if clicked_idx in get_neighbors(blank):
        order = st.session_state.tiles
        order[clicked_idx], order[blank] = order[blank], order[clicked_idx]
        st.session_state.blank_idx = clicked_idx
        st.session_state.moves += 1

        if is_solved(order):
            st.session_state.finished = True


def init_game_from_image(img_bgr):
    prepared = prepare_face_image(img_bgr)
    original_tiles = split_into_tiles(prepared)

    solved_order = get_solved_order()
    shuffled_order, blank_idx = shuffle_safely(solved_order, steps=150)

    st.session_state.captured_image = prepared
    st.session_state.solved_tiles = original_tiles
    st.session_state.tiles = shuffled_order
    st.session_state.blank_idx = blank_idx
    st.session_state.started = True
    st.session_state.start_time = time.time()
    st.session_state.finished = False
    st.session_state.moves = 0


def reset_game():
    st.session_state.captured_image = None
    st.session_state.tiles = None
    st.session_state.blank_idx = None
    st.session_state.solved_tiles = None
    st.session_state.started = False
    st.session_state.start_time = None
    st.session_state.finished = False
    st.session_state.moves = 0


# =========================
# UI
# =========================
st.markdown(
    """
    <h1 style='text-align:center; color:#00e5ff;'>YOUR FACE PUZZLE</h1>
    <p style='text-align:center;'>Ambil foto wajahmu lalu susun puzzlenya secepat mungkin.</p>
    """,
    unsafe_allow_html=True
)

menu = st.radio("Mode", ["Single Player"], horizontal=True)

st.markdown("### 1) Ambil Foto Wajah")
camera_image = st.camera_input("Aktifkan kamera lalu ambil foto")

col_a, col_b = st.columns(2)

with col_a:
    if st.button("Mulai Game", use_container_width=True):
        if camera_image is None:
            st.warning("Silakan ambil foto dulu.")
        else:
            file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            init_game_from_image(img_bgr)
            st.success("Game dimulai!")

with col_b:
    if st.button("Reset", use_container_width=True):
        reset_game()
        st.rerun()

st.markdown("---")

if st.session_state.started and st.session_state.captured_image is not None:
    elapsed = int(time.time() - st.session_state.start_time) if not st.session_state.finished else int(time.time() - st.session_state.start_time)

    m, s = divmod(elapsed, 60)
    st.markdown(f"### ⏱️ Waktu: {m:02d}:{s:02d}")
    st.markdown(f"### 🎯 Moves: {st.session_state.moves}")

    board_img = render_board(st.session_state.tiles, st.session_state.solved_tiles)
    st.image(board_img, caption="Klik nomor posisi di bawah untuk menggeser ubin", use_container_width=False)

    st.markdown("### 2) Geser Puzzle")
    st.caption("Klik posisi ubin yang ingin dipindah. Hanya ubin di sebelah kotak kosong yang bisa bergerak.")

    for r in range(GRID_SIZE):
        cols = st.columns(GRID_SIZE)
        for c in range(GRID_SIZE):
            idx = r * GRID_SIZE + c
            tile_value = st.session_state.tiles[idx]
            label = "⬜" if tile_value == GRID_SIZE * GRID_SIZE - 1 else f"{tile_value + 1}"

            with cols[c]:
                if st.button(label, key=f"btn_{idx}", use_container_width=True):
                    if not st.session_state.finished:
                        move_tile(idx)
                        st.rerun()

    st.markdown("---")
    st.markdown("### 3) Preview Gambar Asli")
    st.image(
        cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB),
        caption="Gambar asli",
        use_container_width=False
    )

    if st.session_state.finished:
        total_time = int(time.time() - st.session_state.start_time)
        mm, ss = divmod(total_time, 60)
        st.success(f"🎉 Selamat! Puzzle selesai dalam {mm:02d}:{ss:02d} dengan {st.session_state.moves} moves.")

else:
    st.info("Ambil foto terlebih dahulu, lalu tekan **Mulai Game**.")

st.markdown("---")
st.markdown(
    """
    **Aturan singkat:**
    - Ambil foto wajah dari kamera
    - Sistem akan memotong gambar menjadi puzzle 3x3
    - Susun kembali sampai benar
    - Ubin hanya bisa bergerak ke kotak kosong
    """
)
