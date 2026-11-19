"""
clipping.studio.gameplay_layout — "gameplay" vertical layout (``--layout gameplay``).

Instead of cropping a face-tracked 9:16 slice out of a 16:9 source, the whole
16:9 frame is shown full-width in the middle of the 9:16 canvas:

    ┌──────────────┐
    │  TITLE BAND  │  clip title (big outlined display font)
    ├──────────────┤
    │  FULL 16:9   │  the entire game screen, nothing cropped
    │   GAMEPLAY   │
    ├──────────────┤
    │ CAPTION BAND │  karaoke captions (positioned by subtitles.buat_file_ass)
    └──────────────┘

The bands are filled with a blurred, darkened copy of the gameplay (or black).
"""

from __future__ import annotations

import os
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TITLE_FONT = {
    "nama": "Bangers",
    "file": "Bangers-Regular.ttf",
    "url": "https://cdn.jsdelivr.net/fontsource/fonts/bangers@latest/latin-400-normal.ttf",
}

# Title look: white fill, thick black outline, soft drop shadow (GTA-style).
TITLE_FILL = (255, 255, 255, 255)
TITLE_STROKE = (0, 0, 0, 255)
TITLE_MAX_LINES = 3


def is_gameplay_layout(cfg, rasio: str) -> bool:
    """True when the gameplay layout applies (vertical output only)."""
    return getattr(cfg, "layout", "crop") == "gameplay" and rasio in {"9:16", "3:4", "4:5", "1:1"}


def geometry(out_w: int, out_h: int, src_w: int = 1920, src_h: int = 1080) -> dict:
    """
    Pixel rectangles (x, y, w, h) for the game window and the two bands.

    The game keeps its own aspect ratio at full canvas width, centred
    vertically, so the title and caption bands are the same height.
    """
    src_ratio = (src_w / src_h) if src_w and src_h else 16 / 9
    game_w = out_w
    game_h = int(round(out_w / src_ratio))
    game_h -= game_h % 2
    game_h = min(game_h, out_h)
    y = (out_h - game_h) // 2
    return {
        "game": (0, y, game_w, game_h),
        "title": (0, 0, out_w, y),
        "caption": (0, y + game_h, out_w, out_h - y - game_h),
    }


def _font_path(cfg) -> str | None:
    """Title font file: user-supplied --title-font-file, else the bundled default (downloaded once)."""
    custom = getattr(cfg, "title_font_file", None)
    if custom and os.path.isfile(custom):
        return custom
    path = os.path.join(cfg.font_dir, TITLE_FONT["file"])
    if not (os.path.isfile(path) and os.path.getsize(path) > 1000):
        # Studio modules are loaded by file path (clipping/studio is not a package).
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "clipping_studio_typography_gl", os.path.join(os.path.dirname(__file__), "typography.py"))
        typography = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(typography)
        typography.download_google_font(TITLE_FONT["url"], TITLE_FONT["file"], cfg.font_dir)
    return path if os.path.isfile(path) and os.path.getsize(path) > 1000 else None


def render_title(text: str, band_w: int, band_h: int, cfg) -> np.ndarray | None:
    """
    Draw the title into a transparent BGRA image of the title band.

    Wraps to at most three lines and shrinks the font until it fits.
    Returns None when there is no title or no room.
    """
    text = " ".join(str(text or "").split()).upper()
    if not text or band_h < 40:
        return None
    font_file = _font_path(cfg)
    pad_x = int(band_w * 0.06)
    max_w = band_w - 2 * pad_x
    max_h = int(band_h * 0.78)

    img = Image.new("RGBA", (band_w, band_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    def load(size):
        if font_file:
            return ImageFont.truetype(font_file, size)
        return ImageFont.load_default(size=size)

    def layout(size):
        font = load(size)
        stroke = max(3, size // 11)
        avg_char = max(1.0, draw.textlength("ABCDEFGHIJKLMNOPQRSTUVWXYZ", font=font) / 26)
        lines = textwrap.wrap(text, width=max(4, int(max_w / avg_char)))
        widths = [draw.textlength(line, font=font) + 2 * stroke for line in lines]
        line_h = int(size * 1.08)
        fits = len(lines) <= TITLE_MAX_LINES and max(widths) <= max_w and line_h * len(lines) <= max_h
        return fits, font, stroke, lines, line_h

    min_size = 28
    size = max(min_size, int(band_h * 0.30))
    fits, font, stroke, lines, line_h = layout(size)
    while not fits and size > min_size:
        size = max(min_size, size - 4)
        fits, font, stroke, lines, line_h = layout(size)
    lines = lines[:TITLE_MAX_LINES]  # very long titles: keep the first lines
    total_h = line_h * len(lines)

    y = (band_h - total_h) // 2 + int(size * 0.05)
    shadow = max(3, size // 14)
    for line in lines:
        w = draw.textlength(line, font=font)
        x = (band_w - w) / 2
        draw.text((x + shadow, y + shadow), line, font=font, fill=(0, 0, 0, 150),
                  stroke_width=stroke, stroke_fill=(0, 0, 0, 150))
        draw.text((x, y), line, font=font, fill=TITLE_FILL, stroke_width=stroke, stroke_fill=TITLE_STROKE)
        y += line_h
    rgba = np.array(img)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


def _background(frame: np.ndarray, out_w: int, out_h: int, style: str) -> np.ndarray:
    if style == "black":
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    # Cover-crop the source to the canvas ratio, blur cheaply at low resolution, darken.
    target = out_w / out_h
    if w / h > target:
        cw = int(h * target)
        x = (w - cw) // 2
        crop = frame[:, x : x + cw]
    else:
        ch = int(w / target)
        y = (h - ch) // 2
        crop = frame[y : y + ch, :]
    small = cv2.resize(crop, (max(8, out_w // 12), max(8, out_h // 12)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    bg = cv2.resize(small, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return (bg * 0.38).astype(np.uint8)


def _blend(canvas: np.ndarray, overlay_bgra: np.ndarray, x: int, y: int) -> None:
    oh, ow = overlay_bgra.shape[:2]
    region = canvas[y : y + oh, x : x + ow]
    alpha = overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
    region[:] = (overlay_bgra[:, :, :3] * alpha + region * (1.0 - alpha)).astype(np.uint8)


def compose(frame: np.ndarray, out_w: int, out_h: int, geom: dict, title_bgra=None,
            background: str = "blur", game_frame: np.ndarray | None = None) -> np.ndarray:
    """
    Build one output frame: background, full game window, title overlay.

    ``game_frame`` replaces what is shown in the game window (e.g. a B-roll
    clip) while the background still comes from ``frame``.
    """
    canvas = _background(frame, out_w, out_h, background)
    gx, gy, gw, gh = geom["game"]
    content = frame if game_frame is None else game_frame
    canvas[gy : gy + gh, gx : gx + gw] = cv2.resize(content, (gw, gh), interpolation=cv2.INTER_AREA)
    # Thin dark edge between gameplay and bands.
    cv2.rectangle(canvas, (gx, gy - 2), (gx + gw - 1, gy + gh + 1), (0, 0, 0), 2)
    if title_bgra is not None:
        tx, ty, _, _ = geom["title"]
        _blend(canvas, title_bgra, tx, ty)
    return canvas


def caption_margin_v(cfg, play_res_x: int, play_res_y: int, font_sz: int, source_dim=None) -> int:
    """Bottom margin (ASS MarginV) that centres one caption line in the caption band."""
    sw, sh = source_dim if source_dim else (1920, 1080)
    _, _, _, band_h = geometry(play_res_x, play_res_y, sw, sh)["caption"]
    return max(10, int(band_h / 2 - font_sz * 0.55))
