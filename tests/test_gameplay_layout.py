"""--layout gameplay: geometry, title band, frame composition, caption placement, CLI."""

import importlib.util
import os
import types
from pathlib import Path

import numpy as np
import pytest

from clipping import config

STUDIO = Path(__file__).resolve().parents[1] / "clipping" / "studio"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"test_{name}", STUDIO / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gl = _load("gameplay_layout")


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("GOOGLE_API_KEY", "NVIDIA_API_KEY", "PEXELS_API_KEY", "HF_TOKEN"):
        monkeypatch.setenv(name, "")
    return tmp_path


def test_geometry_keeps_whole_16x9_frame_centered():
    g = gl.geometry(1080, 1920, 1920, 1080)
    assert g["game"] == (0, 656, 1080, 608)
    assert g["title"] == (0, 0, 1080, 656)
    assert g["caption"] == (0, 1264, 1080, 656)
    # 4:3 source still fits full width
    x, y, w, h = gl.geometry(1080, 1920, 1440, 1080)["game"]
    assert (w, h) == (1080, 810) and y == (1920 - 810) // 2


def test_layout_applies_only_to_vertical_output():
    cfg = types.SimpleNamespace(layout="gameplay")
    assert gl.is_gameplay_layout(cfg, "9:16")
    assert not gl.is_gameplay_layout(cfg, "16:9")
    assert not gl.is_gameplay_layout(types.SimpleNamespace(layout="crop"), "9:16")
    assert not gl.is_gameplay_layout(types.SimpleNamespace(), "9:16")


def test_compose_places_game_and_title():
    frame = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    g = gl.geometry(1080, 1920, 1920, 1080)
    title = np.zeros((656, 1080, 4), dtype=np.uint8)
    title[300:320, 100:200] = (0, 0, 255, 255)  # opaque red patch
    out = gl.compose(frame, 1080, 1920, g, title, "black")
    assert out.shape == (1920, 1080, 3)
    assert out[900, 540].tolist() == [200, 200, 200]          # game window
    assert out[310, 150].tolist() == [0, 0, 255]               # title pixel blended in
    assert out[1700, 540].tolist() == [0, 0, 0]                # black caption band
    blurred = gl.compose(frame, 1080, 1920, g, None, "blur")
    assert 0 < blurred[1700, 540, 0] < 200                     # darkened blur fill
    broll = np.full((608, 1080, 3), 50, dtype=np.uint8)
    with_broll = gl.compose(frame, 1080, 1920, g, None, "black", game_frame=broll)
    assert with_broll[900, 540].tolist() == [50, 50, 50]


def test_title_wraps_and_fits_band(tmp_path):
    cfg = types.SimpleNamespace(font_dir=str(tmp_path), title_font_file=None)
    # No network in tests: point at a font that ships with Pillow's default.
    cfg.title_font_file = None
    gl_font = gl._font_path
    gl._font_path = lambda c: None
    try:
        img = gl.render_title("the worst getaway driver in vice city history ever recorded", 1080, 656, cfg)
    finally:
        gl._font_path = gl_font
    assert img.shape == (656, 1080, 4)
    rows = np.where(img[:, :, 3].max(axis=1) > 0)[0]
    assert rows.min() > 0 and rows.max() < 655                 # fully inside the band
    assert gl.render_title("", 1080, 656, cfg) is None


def test_caption_margin_centres_in_bottom_band():
    m = gl.caption_margin_v(None, 1080, 1920, 150, (1920, 1080))
    assert 230 < m < 260                                       # ~656/2 - 0.55*150


def _ass_cfg(tmp_path, layout):
    cfg = config.build_config(["--url", "https://x.test", "--layout", layout, "--font-style", "GTA"])
    cfg.font_dir = str(tmp_path)
    return cfg


def test_ass_captions_move_to_band_uppercase_and_yellow(clean_env):
    subtitles = _load("subtitles")
    seg = [{"start": 1.0, "end": 2.0, "words": [{"word": "go", "start": 1.0, "end": 1.4},
                                                {"word": "fast", "start": 1.4, "end": 2.0}]}]
    path = clean_env / "g.ass"
    subtitles.buat_file_ass(seg, 0.0, 5.0, str(path), "9:16", _ass_cfg(clean_env, "gameplay"),
                            source_dim=(1920, 1080))
    text = path.read_text(encoding="utf-8")
    style = next(l for l in text.splitlines() if l.startswith("Style: Default"))
    assert style.split(",")[1] == "Anton"
    assert int(style.split(",")[2]) == 150 and int(style.split(",")[-2]) > 200
    assert "GO" in text and "FAST" in text and "&H00D4FF&" in text

    crop = clean_env / "c.ass"
    subtitles.buat_file_ass(seg, 0.0, 5.0, str(crop), "9:16", _ass_cfg(clean_env, "crop"),
                            source_dim=(1920, 1080))
    ctext = crop.read_text(encoding="utf-8")
    assert "go" in ctext and "&H00FFFF&" in ctext               # default behaviour unchanged


def test_cli_layout_flags(clean_env):
    cfg = config.build_config(["--url", "https://x.test"])
    assert (cfg.layout, cfg.layout_background, cfg.title_font_file, cfg.no_title) == ("crop", "blur", None, False)
    cfg = config.build_config(["--url", "https://x.test", "--layout", "gameplay", "--layout-background", "black",
                               "--title-font-file", "pricedown.otf", "--no-title", "--font-style", "GTA"])
    assert cfg.layout == "gameplay" and cfg.layout_background == "black" and cfg.no_title
    assert cfg.title_font_file.endswith("pricedown.otf") and cfg.gaya_font_aktif == "GTA"
    assert cfg.daftar_font["GTA"]["utama"]["nama"] == "Anton"


def test_montserrat_preset_uses_correctly_named_file():
    hz = config.DAFTAR_FONT["HORMOZI"]["utama"]
    assert hz["nama"] == "Montserrat" and "fontsource" not in hz["url"]


def test_filter_escaping(monkeypatch):
    helpers = _load("helpers")
    assert helpers.escape_ffmpeg_filter_value("/tmp/a:b/x.ass") == r"/tmp/a\:b/x.ass" or os.name == "nt"
    monkeypatch.setattr(helpers.os, "name", "nt")
    assert helpers.escape_ffmpeg_filter_value(r"C:\Users\me\x.ass") == r"C\\:/Users/me/x.ass"
    monkeypatch.setattr(helpers.os, "name", "posix")
    assert helpers.escape_ffmpeg_filter_value("/tmp/a:b/x.ass") == r"/tmp/a\:b/x.ass"
