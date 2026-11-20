"""
clipping.live — Live Stream Clipping (rolling window)

Records a live stream continuously into short MPEG-TS segments while the
main loop repeatedly:

  1. takes the most recent ``live_window`` seconds of recording,
  2. transcribes it with Whisper,
  3. asks the AI for the best moment(s) in that window,
  4. renders any clip that scores high enough and was not rendered before.

Windows overlap (``live_window - live_step`` seconds) so a moment that falls
on a window boundary is still seen whole by the next window. Clips that end
too close to the edge of a window are deferred to the next window instead
of being cut off.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field

# Max clip length the AI may choose (engine.MAX_CLIP_DURATION). Window overlap
# must be at least this long so a deferred clip fits whole in the next window.
MAX_CLIP_SECONDS = 179

# Gemini limits per window. The normal pipeline waits up to 10 attempts with a
# 15-minute timeout each, which can stall a live run for 40+ minutes when the
# model is overloaded. In live mode a failed window is cheap: the next window
# overlaps it, so give up quickly. Worst case ~2 x 150 s + 10 s + fallback 150 s.
LIVE_GEMINI_TIMEOUT_S = 150
LIVE_GEMINI_MAX_ATTEMPTS = 2
LIVE_GEMINI_RETRY_WAIT_S = 10


# ==============================================================================
# Pure helpers (unit-tested)
# ==============================================================================

@dataclass
class Segment:
    path: str
    start: float      # seconds since recording began (recorded timeline)
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def parse_segment_list(csv_text: str) -> list[tuple[str, float, float]]:
    """
    Parse an ffmpeg ``-segment_list_type csv`` list.

    Each completed segment is one ``filename,start,end`` row. Rows that are
    incomplete (ffmpeg still writing the line) are ignored.
    """
    rows = []
    for row in csv.reader(csv_text.splitlines()):
        if len(row) != 3:
            continue
        name, start, end = row
        try:
            s, e = float(start), float(end)
        except ValueError:
            continue
        if e > s:
            rows.append((name, s, e))
    return rows


def select_window(segments: list[Segment], window_seconds: float) -> list[Segment]:
    """Return the trailing segments that cover the last ``window_seconds``."""
    if not segments:
        return []
    end = segments[-1].end
    cutoff = end - window_seconds
    return [s for s in segments if s.end > cutoff]


def segments_to_prune(segments: list[Segment], keep_after: float) -> list[Segment]:
    """Segments that end before ``keep_after`` and are no longer needed."""
    return [s for s in segments if s.end <= keep_after]


class ClipDeduper:
    """Remembers rendered clip ranges (recorded timeline) to skip repeats."""

    def __init__(self, max_overlap: float = 0.5):
        self.max_overlap = max_overlap
        self.ranges: list[tuple[float, float]] = []

    def is_duplicate(self, start: float, end: float) -> bool:
        length = end - start
        if length <= 0:
            return True
        for s, e in self.ranges:
            overlap = min(end, e) - max(start, s)
            if overlap <= 0:
                continue
            shorter = min(length, e - s)
            if overlap / shorter > self.max_overlap:
                return True
        return False

    def add(self, start: float, end: float) -> None:
        self.ranges.append((start, end))


def make_clip_filter(
    window_start: float,
    window_duration: float,
    deduper: ClipDeduper,
    min_score: int,
    max_clips: int,
    edge_guard: float,
    is_final: bool,
    log=print,
):
    """
    Build a ``clip_filter`` for ``runner.analyze_and_render``.

    Keeps clips that score ``>= min_score``, are not duplicates of clips
    already rendered, and (unless this is the final window) end at least
    ``edge_guard`` seconds before the end of the window. Accepted clips are
    registered with the deduper, renumbered 1..n and capped at ``max_clips``.
    """

    def _filter(clips: list[dict]) -> list[dict]:
        kept = []
        for clip in sorted(clips, key=lambda c: -int(c.get("viral_score", 0) or 0)):
            try:
                start = float(clip["start_time"])
                end = float(clip["end_time"])
            except (KeyError, TypeError, ValueError):
                continue
            score = int(clip.get("viral_score", 0) or 0)
            abs_start, abs_end = window_start + start, window_start + end
            label = f"{_hms(abs_start)}-{_hms(abs_end)} (score {score})"

            if score < min_score:
                log(f"   ⏭️ Skip {label}: below --live-min-score {min_score}")
                continue
            if not is_final and end > window_duration - edge_guard:
                log(f"   ⏳ Defer {label}: too close to the live edge, next window will see it whole")
                continue
            if deduper.is_duplicate(abs_start, abs_end):
                log(f"   ⏭️ Skip {label}: already clipped")
                continue

            deduper.add(abs_start, abs_end)
            clip["live_stream_start"] = round(abs_start, 2)
            clip["live_stream_end"] = round(abs_end, 2)
            kept.append(clip)
            if len(kept) >= max_clips:
                break

        for i, clip in enumerate(kept, start=1):
            clip["rank"] = i
        _filter.accepted = list(kept)
        return kept

    _filter.accepted = []
    return _filter


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def build_record_cmd(
    inputs: list[tuple[str, dict]],
    out_dir: str,
    run_id: int,
    start_number: int,
    segment_seconds: int,
    realtime_input: bool = False,
) -> list[str]:
    """
    ffmpeg command that copies a stream into numbered .ts segments.

    ``inputs`` is ``[(url, http_headers)]`` for a combined audio+video stream,
    or ``[(video_url, h), (audio_url, h)]`` when video and audio are separate
    (YouTube live).
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    for url, headers in inputs:
        if headers:
            header_blob = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
            cmd += ["-headers", header_blob]
        if realtime_input:
            cmd += ["-re"]
        cmd += ["-i", url]
    if len(inputs) > 1:
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        cmd += ["-map", "0:v:0?", "-map", "0:a:0?"]
    cmd += [
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(segment_seconds),
        "-segment_start_number", str(start_number),
        "-reset_timestamps", "1",
        "-segment_list", os.path.join(out_dir, f"segments_{run_id:03d}.csv"),
        "-segment_list_type", "csv",
        "-segment_list_flags", "+live",
        os.path.join(out_dir, "seg_%06d.ts"),
    ]
    return cmd


def build_concat_cmd(list_file: str, out_path: str) -> list[str]:
    """ffmpeg command that joins .ts segments into one mp4 without re-encoding."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "concat", "-safe", "0", "-i", list_file,
        "-c", "copy", "-bsf:a", "aac_adtstoasc",
        "-movflags", "+faststart",
        out_path,
    ]


# ==============================================================================
# Stream resolution & recording
# ==============================================================================

def resolve_live_stream(url: str, cookies: str | None = None, max_height="max") -> dict:
    """
    Ask yt-dlp for the stream's current HLS URL.

    Returns ``{"is_live", "live_status", "title", "inputs"}`` where
    ``inputs`` is ``[(url, headers), ...]`` for :func:`build_record_cmd`.
    """
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError
    from .engine import _apply_cookies

    h = "" if max_height in (None, "max") else f"[height<=?{int(max_height)}]"
    hls = "[protocol*=m3u8]"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # YouTube live: separate H.264 video + audio HLS. Others: combined HLS.
        "format": (
            f"bestvideo{h}{hls}[vcodec!*=av01]+bestaudio{hls}/"
            f"best{h}{hls}/best{hls}/bestvideo{h}+bestaudio/best"
        ),
        "remote_components": ["ejs:github"],
    }
    _apply_cookies(opts, cookies)
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as e:
        msg = str(e).replace("ERROR: ", "")
        raise RuntimeError(f"Could not open stream: {msg}") from None

    live_status = info.get("live_status") or ("is_live" if info.get("is_live") else "not_live")
    return {
        "is_live": live_status == "is_live",
        "live_status": live_status,
        "title": info.get("title") or "",
        "inputs": stream_inputs(info),
    }


def stream_inputs(info: dict) -> list[tuple[str, dict]]:
    """Extract ``[(url, headers)]`` from a yt-dlp info dict."""
    if info.get("requested_formats"):
        return [
            (f["url"], f.get("http_headers") or {})
            for f in info["requested_formats"] if f.get("url")
        ]
    if info.get("url"):
        return [(info["url"], info.get("http_headers") or {})]
    return []


@dataclass
class LiveRecorder:
    """
    Keeps an ffmpeg process copying the live stream into segments.

    ffmpeg is restarted when it exits while the stream is still live (stream
    URLs expire after a few hours, and network hiccups happen). The recorded
    timeline is the sum of recorded segment durations, so any gap during a
    restart is simply missing from the recording.
    """

    url: str
    out_dir: str
    segment_seconds: int = 10
    cookies: str | None = None
    max_height: object = "max"
    local_input: bool = False          # treat url as a local file/URL (tests, replays)
    restart_check_interval: float = 30.0
    log: object = print

    segments: list[Segment] = field(default_factory=list)
    ended: bool = False
    title: str = ""
    _proc: subprocess.Popen | None = None
    _run_id: int = 0
    _seen: set = field(default_factory=set)
    _needs_restart: bool = False
    _last_restart_check: float = 0.0

    def start(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        if self.local_input:
            self._launch([(self.url, {})], realtime=True)
            return
        info = resolve_live_stream(self.url, self.cookies, self.max_height)
        self.title = info["title"]
        if not info["is_live"]:
            raise RuntimeError(
                f"Stream is not live right now (live_status={info['live_status']}). "
                "For a finished stream, run without --live."
            )
        if not info["inputs"]:
            raise RuntimeError("yt-dlp did not return a stream URL for this live stream.")
        self._launch(info["inputs"])

    def _launch(self, inputs: list[tuple[str, dict]], realtime: bool = False) -> None:
        self._run_id += 1
        next_number = len(self._seen)
        cmd = build_record_cmd(
            inputs, self.out_dir, self._run_id, next_number,
            self.segment_seconds, realtime_input=realtime,
        )
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def poll(self) -> list[Segment]:
        """Pick up newly completed segments and restart ffmpeg if needed."""
        new = []
        for run in range(1, self._run_id + 1):
            list_path = os.path.join(self.out_dir, f"segments_{run:03d}.csv")
            if not os.path.exists(list_path):
                continue
            with open(list_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            for name, s, e in parse_segment_list(text):
                if name in self._seen:
                    continue
                self._seen.add(name)
                start = self.segments[-1].end if self.segments else 0.0
                seg = Segment(os.path.join(self.out_dir, name), start, e - s)
                self.segments.append(seg)
                new.append(seg)

        if self._proc is not None and self._proc.poll() is not None and not self.ended:
            self._on_exit()
        if self._needs_restart and not self.ended:
            self._try_restart()
        return new

    def _on_exit(self) -> None:
        err = b""
        try:
            err = self._proc.stderr.read() if self._proc.stderr else b""
        except Exception:
            pass
        code = self._proc.returncode
        self._proc = None
        if self.local_input:
            self.ended = True
            return
        tail = err.decode("utf-8", "replace").strip().splitlines()[-1:] if err else []
        self.log(f"🔁 Recorder exited (code {code}{': ' + tail[0] if tail else ''}); checking stream...")
        self._needs_restart = True

    def _try_restart(self) -> None:
        """Re-resolve the stream; reconnect if still live, else mark ended."""
        now = time.time()
        if now - self._last_restart_check < self.restart_check_interval:
            return
        self._last_restart_check = now
        try:
            info = resolve_live_stream(self.url, self.cookies, self.max_height)
        except Exception as e:
            self.log(f"⚠️ Stream check failed ({e}); retrying in {self.restart_check_interval:.0f}s...")
            return
        self._needs_restart = False
        if not info["is_live"] or not info["inputs"]:
            self.log("📴 Stream has ended.")
            self.ended = True
            return
        self.log("🔁 Reconnected to stream.")
        self._launch(info["inputs"])

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                # 'q' is not available (-nostdin); terminate lets ffmpeg close the segment.
                self._proc.terminate()
                self._proc.wait(timeout=15)
            except Exception:
                self._proc.kill()
        self._proc = None

    @property
    def recorded_seconds(self) -> float:
        return self.segments[-1].end if self.segments else 0.0


def build_window_video(segments: list[Segment], out_path: str) -> str:
    """Join segments into one mp4 (stream copy, takes seconds)."""
    list_file = out_path + ".txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for seg in segments:
            safe = os.path.abspath(seg.path).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    result = subprocess.run(build_concat_cmd(list_file, out_path), capture_output=True, text=True)
    os.remove(list_file)
    if result.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"Failed to build window video: {result.stderr.strip()[-500:]}")
    return out_path


# ==============================================================================
# Main loop
# ==============================================================================

def validate_live_settings(window: float, step: float, segment: float) -> list[str]:
    """Return warnings about settings that can lose or cut off clips."""
    warnings = []
    if step > window:
        warnings.append(
            f"--live-step ({step:.0f}s) is longer than --live-window ({window:.0f}s): "
            "parts of the stream will never be analysed."
        )
    elif window - step < MAX_CLIP_SECONDS:
        warnings.append(
            f"Window overlap is {window - step:.0f}s (< {MAX_CLIP_SECONDS}s max clip length): "
            "long moments near a window edge may be missed."
        )
    if segment > step:
        warnings.append("--live-segment is longer than --live-step; windows will lag behind.")
    return warnings


def run_live(cfg, recorder: LiveRecorder | None = None, sleep=time.sleep, clock=time.time) -> list[dict]:
    """
    Clip a live stream while it is live. Runs until the stream ends,
    ``cfg.live_max_hours`` passes, or Ctrl+C. Returns the live manifest.
    """
    from . import engine, runner

    window_s = float(cfg.live_window)
    step_s = float(cfg.live_step)
    edge_guard = float(getattr(cfg, "live_edge_guard", 10))
    for w in validate_live_settings(window_s, step_s, cfg.live_segment):
        print(f"⚠️ {w}")

    url = cfg.url_youtube if isinstance(cfg.url_youtube, str) else cfg.url_youtube[0]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    live_dir = os.path.join(cfg.outputs_dir, "live", stamp)
    seg_dir = os.path.join(live_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)
    manifest_path = os.path.join(live_dir, "live_manifest.json")

    if recorder is None:
        recorder = LiveRecorder(
            url=url,
            out_dir=seg_dir,
            segment_seconds=int(cfg.live_segment),
            cookies=getattr(cfg, "yt_cookies", None),
            max_height=getattr(cfg, "download_source_height", "max"),
        )

    # Load Whisper once and reuse it for every window.
    whisper_model = None
    try:
        from faster_whisper import WhisperModel
        print(f"⏳ Loading Whisper '{cfg.whisper_model}' ({cfg.whisper_device})...", flush=True)
        whisper_model = WhisperModel(
            cfg.whisper_model, device=cfg.whisper_device, compute_type=cfg.whisper_compute_type
        )
    except Exception as e:
        print(f"⚠️ Could not pre-load Whisper ({e}); it will load per window.")

    print(f"🔴 Connecting to live stream: {url}")
    recorder.start()
    if recorder.title:
        print(f"   Title: {recorder.title}")

    uploader = None
    drive_run_name = f"{time.strftime('%Y-%m-%d %H.%M')} {recorder.title or 'live'}"
    if getattr(cfg, "upload_drive", False):
        try:
            from .drive_upload import DriveUploader
            uploader = DriveUploader.from_env()
            uploader.run_folder(drive_run_name)
            print(f"   Drive : uploading clips to '{uploader.root_folder}/{drive_run_name}'")
        except Exception as e:
            uploader = None
            print(f"   ⚠️ Google Drive upload disabled: {e}")
    print(f"   Output: {live_dir}")
    print(
        f"   Window {window_s:.0f}s, every {step_s:.0f}s, min score {cfg.live_min_score}, "
        f"up to {cfg.live_clips_per_window} clip(s) per window. Ctrl+C to stop.",
        flush=True,
    )

    deduper = ClipDeduper()
    manifest: list[dict] = []
    processed_until = 0.0
    window_index = 0
    started = clock()
    max_seconds = float(getattr(cfg, "live_max_hours", 0) or 0) * 3600
    stop_reason = "stream ended"

    def _save_manifest():
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    try:
        while True:
            recorder.poll()
            recorded = recorder.recorded_seconds
            pending = recorded - processed_until
            final = recorder.ended

            if max_seconds and clock() - started >= max_seconds:
                stop_reason = f"--live-max-hours {cfg.live_max_hours} reached"
                recorder.stop()
                final = True

            due = pending >= step_s or (final and pending >= 20)
            if not due:
                if final:
                    break
                sleep(2)
                continue

            window_segments = select_window(recorder.segments, window_s)
            win_start = window_segments[0].start
            win_end = window_segments[-1].end
            if processed_until and win_start > processed_until:
                print(
                    f"⚠️ Fell behind: {_hms(processed_until)}-{_hms(win_start)} of the stream "
                    "was not analysed. Use a GPU, a smaller --whisper-model, or a larger --live-step."
                )

            window_index += 1
            win_dir = os.path.join(live_dir, f"window_{window_index:04d}")
            os.makedirs(win_dir, exist_ok=True)
            print(f"\n{'━' * 70}")
            print(
                f"🔴 Window {window_index}: stream {_hms(win_start)} → {_hms(win_end)} "
                f"({win_end - win_start:.0f}s)",
                flush=True,
            )
            t0 = clock()
            processed_until = win_end

            try:
                win_video = build_window_video(window_segments, os.path.join(win_dir, "window.mp4"))
                win_cfg = copy.copy(cfg)
                win_cfg.file_video_asli = win_video
                win_cfg.outputs_dir = win_dir
                win_cfg.jumlah_clip = cfg.live_clips_per_window
                win_cfg.use_dlp_subs = False
                win_cfg.load_gemini_json = False
                win_cfg.gemini_timeout_s = LIVE_GEMINI_TIMEOUT_S
                win_cfg.gemini_max_attempts = LIVE_GEMINI_MAX_ATTEMPTS
                win_cfg.gemini_retry_wait_s = LIVE_GEMINI_RETRY_WAIT_S
                win_cfg.gemini_retry_increment_s = 0

                transcript, segs = engine.transcribe_video(
                    win_video,
                    max_words_per_subtitle=cfg.max_kata_per_subtitle,
                    model_size=cfg.whisper_model,
                    device=cfg.whisper_device,
                    compute_type=cfg.whisper_compute_type,
                    whisper_model_instance=whisper_model,
                )
                if not transcript.strip():
                    print("   🤫 No speech in this window, skipping AI analysis.")
                else:
                    clip_filter = make_clip_filter(
                        window_start=win_start,
                        window_duration=win_end - win_start,
                        deduper=deduper,
                        min_score=int(cfg.live_min_score),
                        max_clips=int(cfg.live_clips_per_window),
                        edge_guard=edge_guard,
                        is_final=final,
                    )
                    rendered = runner.analyze_and_render(win_cfg, transcript, segs, clip_filter=clip_filter)
                    by_rank = {c["rank"]: c for c in clip_filter.accepted}
                    for row in rendered:
                        clip = by_rank.get(row.get("rank"), {})
                        row["live_window"] = window_index
                        row["live_stream_start"] = clip.get("live_stream_start")
                        row["live_stream_end"] = clip.get("live_stream_end")
                        row["source_url"] = url
                    manifest.extend(rendered)
                    _save_manifest()
                    for row in rendered:
                        at = _hms(row["live_stream_start"] or 0)
                        if row.get("status") == "success":
                            print(f"   ✅ Clip ready (stream {at}): {row.get('video_path')}")
                            if uploader is not None:
                                row["drive_link"] = uploader.upload_clip(row, drive_run_name)
                        else:
                            print(f"   ❌ Clip at stream {at} failed to render: {row.get('error')}")
                    if uploader is not None:
                        _save_manifest()
                    if not rendered:
                        print("   (no new clip in this window)")
                if os.path.exists(win_video):
                    os.remove(win_video)
            except Exception as e:
                print(f"   ❌ Window {window_index} failed: {e}. Continuing with the next window.")

            # Free disk: drop segments no future window can include.
            keep_after = recorder.recorded_seconds - window_s - step_s
            for seg in segments_to_prune(recorder.segments, keep_after):
                try:
                    os.remove(seg.path)
                except OSError:
                    pass

            recorder.poll()  # count segments recorded while this window was processed
            lag = recorder.recorded_seconds - win_end
            print(f"   ⏱️ Window took {clock() - t0:.0f}s; now {lag:.0f}s behind live.", flush=True)

            if final and recorder.recorded_seconds - processed_until < 20:
                break
    except KeyboardInterrupt:
        stop_reason = "stopped by user"
        print("\n🛑 Stopping...")
    finally:
        recorder.stop()
        _save_manifest()

    print(f"\n{'=' * 70}")
    print(f"🔴 Live clipping finished ({stop_reason}).")
    print(f"   Recorded : {_hms(recorder.recorded_seconds)}")
    print(f"   Clips    : {len(manifest)}")
    print(f"   Manifest : {manifest_path}")
    print(f"{'=' * 70}")
    if getattr(cfg, "cleanup_source", False):
        shutil.rmtree(seg_dir, ignore_errors=True)
    return manifest
