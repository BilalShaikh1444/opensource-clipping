"""Live stream mode: rolling-window selection, filtering, CLI and loop behaviour."""

import json
import os
import shutil
import subprocess
import sys
import types

import pytest

from clipping import config, live, runner
from clipping.live import ClipDeduper, Segment


def _segments(n, dur=10.0, prefix="seg"):
    return [Segment(f"{prefix}_{i:06d}.ts", i * dur, dur) for i in range(n)]


# ── Pure helpers ──────────────────────────────────────────────────────────────

def test_parse_segment_list_skips_partial_and_bad_rows():
    text = "seg_000000.ts,0.000000,10.010000\nseg_000001.ts,10.010000,20.0\nseg_000002.ts,20.0"
    assert live.parse_segment_list(text) == [
        ("seg_000000.ts", 0.0, 10.01),
        ("seg_000001.ts", 10.01, 20.0),
    ]
    assert live.parse_segment_list("a.ts,x,y\nb.ts,5,5\n") == []


def test_select_window_returns_trailing_segments():
    segs = _segments(100)  # 1000s recorded
    win = live.select_window(segs, 600)
    assert win[0].start == 400 and win[-1].end == 1000
    assert live.select_window(segs[:5], 600) == segs[:5]  # shorter than window
    assert live.select_window([], 600) == []


def test_prune_keeps_segments_needed_by_future_windows():
    segs = _segments(100)
    pruned = live.segments_to_prune(segs, keep_after=100)
    assert [s.end for s in pruned] == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def test_deduper_rejects_mostly_overlapping_clips():
    d = ClipDeduper()
    d.add(100, 160)
    assert d.is_duplicate(110, 170)       # 50/60 overlap
    assert not d.is_duplicate(150, 230)   # 10s overlap only
    assert not d.is_duplicate(200, 260)
    assert d.is_duplicate(50, 50)         # empty range


def _clip(start, end, score):
    return {"rank": 9, "start_time": start, "end_time": end, "viral_score": score}


def test_clip_filter_applies_score_edge_dedupe_and_cap():
    d = ClipDeduper()
    d.add(1000 + 10, 1000 + 70)  # already rendered in stream time
    logs = []
    f = live.make_clip_filter(
        window_start=1000, window_duration=600, deduper=d, min_score=80,
        max_clips=2, edge_guard=10, is_final=False, log=logs.append,
    )
    clips = [
        _clip(20, 60, 99),     # duplicate of rendered clip
        _clip(100, 150, 70),   # below min score
        _clip(500, 595, 95),   # ends inside edge guard -> deferred
        _clip(200, 260, 85),   # lowest passing score -> dropped by the cap of 2
        _clip(300, 360, 90),   # kept, rank 1
        _clip(400, 450, 88),   # kept, rank 2
    ]
    kept = f(clips)
    assert [(c["start_time"], c["rank"]) for c in kept] == [(300, 1), (400, 2)]
    assert kept[0]["live_stream_start"] == 1300 and kept[0]["live_stream_end"] == 1360
    assert f.accepted == kept
    assert d.is_duplicate(1300, 1360)
    assert any("Defer" in m for m in logs) and any("already clipped" in m for m in logs)
    low = live.make_clip_filter(0, 600, ClipDeduper(), 80, 3, 10, False, log=logs.append)
    assert low([_clip(100, 150, 70)]) == [] and "below" in logs[-1]


def test_clip_filter_final_window_keeps_edge_clips():
    f = live.make_clip_filter(0, 600, ClipDeduper(), 80, 1, 10, is_final=True, log=lambda m: None)
    assert len(f([_clip(500, 598, 90)])) == 1


def test_validate_live_settings_warns_on_short_overlap_and_gaps():
    assert live.validate_live_settings(600, 300, 10) == []
    assert "overlap" in live.validate_live_settings(300, 200, 10)[0]
    assert "never be analysed" in live.validate_live_settings(300, 400, 10)[0]


def test_record_and_concat_commands():
    cmd = live.build_record_cmd([("https://x/m3u8", {"User-Agent": "UA"})], "out", 2, 42, 10)
    assert cmd[cmd.index("-segment_start_number") + 1] == "42"
    assert cmd[cmd.index("-segment_list") + 1].endswith("segments_002.csv")
    assert cmd[cmd.index("-headers") + 1] == "User-Agent: UA\r\n"
    assert "-re" not in cmd and cmd[cmd.index("-c") + 1] == "copy"
    assert "0:a:0?" in cmd
    split = live.build_record_cmd([("https://v", {}), ("https://a", {})], "out", 1, 0, 10)
    assert split.count("-i") == 2 and "1:a:0" in split and "-headers" not in split
    concat = live.build_concat_cmd("list.txt", "w.mp4")
    assert concat[-1] == "w.mp4" and "aac_adtstoasc" in concat


def test_stream_inputs_handles_split_and_combined_formats():
    split = {"requested_formats": [
        {"url": "https://v", "http_headers": {"A": "1"}},
        {"url": "https://a"},
    ]}
    assert live.stream_inputs(split) == [("https://v", {"A": "1"}), ("https://a", {})]
    assert live.stream_inputs({"url": "https://c"}) == [("https://c", {})]
    assert live.stream_inputs({}) == []


# ── CLI ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("GOOGLE_API_KEY", "NVIDIA_API_KEY", "PEXELS_API_KEY", "HF_TOKEN"):
        monkeypatch.setenv(name, "")
    return tmp_path


def test_cli_live_defaults_and_overrides(clean_env):
    cfg = config.build_config(["--url", "https://example.test/live"])
    assert cfg.live is False
    cfg = config.build_config([
        "--url", "https://example.test/live", "--live", "--live-window", "900",
        "--live-step", "300", "--live-min-score", "75", "--live-clips-per-window", "2",
        "--live-max-hours", "1.5",
    ])
    assert (cfg.live, cfg.live_window, cfg.live_step, cfg.live_segment) == (True, 900, 300, 10)
    assert (cfg.live_min_score, cfg.live_clips_per_window, cfg.live_max_hours) == (75, 2, 1.5)


@pytest.mark.parametrize("argv", [
    ["--url", "https://a.test", "https://b.test", "--live"],
    ["--url", "https://a.test", "--live", "--live-min-score", "0"],
    ["--url", "https://a.test", "--live", "--live-step", "5"],
    ["--story-mode", "--live"],
])
def test_cli_live_rejects_invalid_settings(clean_env, argv):
    with pytest.raises(SystemExit):
        config.build_config(argv)


# ── Runner integration ────────────────────────────────────────────────────────

def test_analyze_and_render_skips_render_when_filter_rejects_all(clean_env, monkeypatch):
    from clipping import engine, studio

    cfg = config.build_config(["--url", "https://example.test/v"])
    monkeypatch.setattr(engine, "analyze_with_ai", lambda t, c: [_clip(1, 30, 50)])
    monkeypatch.setattr(runner.metadata, "normalize_and_validate", lambda x: x)
    monkeypatch.setattr(runner.metadata, "print_preview", lambda x: None)
    monkeypatch.setattr(runner.metadata, "save_metadata_preview", lambda x, path: None)
    monkeypatch.setattr(studio, "proses_klip", lambda *a, **k: pytest.fail("must not render"))
    assert runner.analyze_and_render(cfg, "[0 - 1] hi", [], clip_filter=lambda c: []) == []


class FakeRecorder:
    """Adds 100s of recording per poll; ends after `total` seconds."""

    def __init__(self, total):
        self.total, self.segments, self.ended, self.title = total, [], False, "Fake stream"
        self.stopped = False

    def start(self):
        pass

    def poll(self):
        if self.recorded_seconds < self.total:
            self.segments += [Segment("x.ts", self.recorded_seconds + i * 10, 10) for i in range(10)]
        else:
            self.ended = True

    def stop(self):
        self.stopped = True

    @property
    def recorded_seconds(self):
        return self.segments[-1].end if self.segments else 0.0


def test_run_live_loop_renders_dedupes_and_writes_manifest(clean_env, monkeypatch):
    from clipping import engine

    cfg = config.build_config(["--url", "https://example.test/live", "--live", "--whisper-device", "cpu"])
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=lambda *a, **k: "model"))
    monkeypatch.setattr(live, "build_window_video", lambda segs, out: out)
    monkeypatch.setattr(live, "segments_to_prune", lambda segs, keep_after: [])
    monkeypatch.setattr(
        engine, "transcribe_video", lambda path, **k: ("[0.0 - 5.0] something happened", [])
    )

    windows = []

    def fake_analyze_and_render(win_cfg, transcript, segs, clip_filter):
        windows.append(win_cfg.outputs_dir)
        # The AI keeps proposing the same moment (stream 250-310s) + a weak one.
        win_start = 0 if len(windows) == 1 else (len(windows) - 1) * 300 - 300
        kept = clip_filter([_clip(250 - win_start, 310 - win_start, 92), _clip(10, 40, 60)])
        return [{"rank": c["rank"], "status": "success", "video_path": "clip.mp4"} for c in kept]

    monkeypatch.setattr(runner, "analyze_and_render", fake_analyze_and_render)

    manifest = live.run_live(cfg, recorder=FakeRecorder(total=1200), sleep=lambda s: None)

    assert len(windows) == 4            # every 300s of a 1200s stream
    assert len(manifest) == 1           # same moment never rendered twice
    row = manifest[0]
    # Window 1 (0-300s) defers it (ends near the live edge); window 2 renders it.
    assert (row["live_stream_start"], row["live_stream_end"], row["live_window"]) == (250, 310, 2)
    saved = json.loads(open(os.path.join(os.path.dirname(windows[0]), "live_manifest.json")).read())
    assert saved == manifest


def test_run_live_keeps_going_when_a_window_fails(clean_env, monkeypatch):
    from clipping import engine

    cfg = config.build_config(["--url", "https://example.test/live", "--live"])
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=lambda *a, **k: None))
    monkeypatch.setattr(live, "build_window_video", lambda segs, out: out)
    calls = []

    def flaky(path, **k):
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return ("", [])  # silence -> no AI call

    monkeypatch.setattr(engine, "transcribe_video", flaky)
    monkeypatch.setattr(runner, "analyze_and_render", lambda *a, **k: pytest.fail("silent window"))
    rec = FakeRecorder(total=600)
    assert live.run_live(cfg, recorder=rec, sleep=lambda s: None) == []
    assert len(calls) == 2 and rec.stopped


# ── Real ffmpeg (skipped when ffmpeg is unavailable) ─────────────────────────

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_recorder_segments_and_window_join_with_real_ffmpeg(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-c:v", "libx264", "-g", "25", "-c:a", "aac", "-shortest", str(src)],
        check=True,
    )
    rec = live.LiveRecorder(url=str(src), out_dir=str(tmp_path / "segs"), segment_seconds=2, local_input=True)
    rec.start()
    import time
    deadline = time.time() + 30
    while not rec.ended and time.time() < deadline:
        rec.poll()
        time.sleep(0.5)
    rec.poll()
    assert rec.ended
    assert len(rec.segments) >= 2
    assert 5.0 <= rec.recorded_seconds <= 7.0

    out = live.build_window_video(live.select_window(rec.segments, 4), str(tmp_path / "w.mp4"))
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
        capture_output=True, text=True, check=True,
    )
    assert 3.5 <= float(probe.stdout.strip()) <= 6.5
