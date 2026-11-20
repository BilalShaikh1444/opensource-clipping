"""Google Drive upload of live clips (fake Drive service, no network)."""

import json
import sys
import types

import pytest

from clipping import config, drive_upload, live, runner
from clipping.drive_upload import DriveUploader


class FakeRequest:
    def __init__(self, result):
        self.result = result

    def execute(self):
        return self.result

    def next_chunk(self):
        return None, self.result


class FakeFiles:
    def __init__(self, existing=None, fail_on=None):
        self.created, self.queries = [], []
        self.existing = existing or {}
        self.fail_on = fail_on

    def list(self, q, fields, pageSize):
        self.queries.append(q)
        hit = [fid for name, fid in self.existing.items() if f"name = '{name}'" in q]
        return FakeRequest({"files": [{"id": hit[0]}] if hit else []})

    def create(self, body, fields, media_body=None):
        if self.fail_on and self.fail_on in body["name"]:
            raise RuntimeError("quota exceeded")
        fid = f"id{len(self.created)}"
        self.created.append({"body": body, "media": media_body})
        return FakeRequest({"id": fid, "webViewLink": f"https://drive/{fid}"})


def _service(files):
    return types.SimpleNamespace(files=lambda: files)


@pytest.fixture(autouse=True)
def fake_media(monkeypatch):
    import googleapiclient.http as http

    monkeypatch.setattr(http, "MediaFileUpload",
                        lambda path, mimetype=None, resumable=True, chunksize=0: ("file", path, mimetype))
    monkeypatch.setattr(http, "MediaInMemoryUpload",
                        lambda data, mimetype=None, resumable=False: ("text", data.decode(), mimetype))


def _row(tmp_path, **kw):
    video = tmp_path / "highlight_rank_1_ready.mp4"
    video.write_bytes(b"video")
    thumb = tmp_path / "thumbnail_rank_1.jpg"
    thumb.write_bytes(b"jpg")
    row = {"status": "success", "video_path": str(video), "thumbnail_path": str(thumb),
           "live_stream_start": 3790.4, "youtube_title_final": 'NPC roasts: "skill issue"?',
           "youtube_description_final": "desc #gta6", "youtube_tags_final": ["gta 6", "fails"],
           "tiktok_caption_final": "cap", "source_url": "https://yt/live"}
    row.update(kw)
    return row


def test_upload_clip_creates_folders_and_three_files(tmp_path):
    files = FakeFiles()
    up = DriveUploader(_service(files), log=lambda m: None)
    link = up.upload_clip(_row(tmp_path), "2026-11-20 14.05 GTA 6 stream")
    names = [c["body"]["name"] for c in files.created]
    assert names[:2] == ["Live Clips", "2026-11-20 14.05 GTA 6 stream"]        # root + run folder
    assert names[2:] == ['1h03m10s - NPC roasts skill issue.mp4',      # unsafe :"? removed
                         '1h03m10s - NPC roasts skill issue.jpg',
                         '1h03m10s - NPC roasts skill issue.txt']
    run_folder = files.created[1]["body"]
    assert run_folder["parents"] == ["id0"] and files.created[2]["body"]["parents"] == ["id1"]
    info = files.created[4]["media"][1]
    assert "TITLE:" in info and "gta 6, fails" in info and "1:03:10" in info
    assert link == "https://drive/id2"
    # Folders are cached: a second clip does not recreate them.
    up.upload_clip(_row(tmp_path, live_stream_start=10), "2026-11-20 14.05 GTA 6 stream")
    assert [c["body"]["name"] for c in files.created].count("Live Clips") == 1


def test_existing_folder_is_reused(tmp_path):
    files = FakeFiles(existing={"Live Clips": "root42"})
    up = DriveUploader(_service(files), log=lambda m: None)
    assert up.folder("Live Clips") == "root42" and files.created == []


def test_failures_never_raise_and_skip_unrendered(tmp_path):
    logs = []
    up = DriveUploader(_service(FakeFiles(fail_on=".mp4")), log=logs.append)
    assert up.upload_clip(_row(tmp_path), "run") is None
    assert "upload failed" in logs[-1]
    assert up.upload_clip(_row(tmp_path, status="failed"), "run") is None
    assert up.upload_clip(_row(tmp_path, video_path=str(tmp_path / "missing.mp4")), "run") is None


def test_from_env_validates_token():
    with pytest.raises(RuntimeError, match="not set"):
        DriveUploader.from_env({})
    with pytest.raises(RuntimeError, match="not valid JSON"):
        DriveUploader.from_env({"GDRIVE_TOKEN_JSON": "{oops"})
    with pytest.raises(RuntimeError, match="refresh_token"):
        DriveUploader.from_env({"GDRIVE_TOKEN_JSON": json.dumps({"client_id": "x"})})
    token = {"refresh_token": "r", "client_id": "c", "client_secret": "s", "token": "t",
             "token_uri": "https://oauth2.googleapis.com/token"}
    up = DriveUploader.from_env({"GDRIVE_TOKEN_JSON": json.dumps(token), "GDRIVE_FOLDER": "GTA Clips"})
    assert up.root_folder == "GTA Clips"


def test_live_uploads_each_successful_clip(monkeypatch, tmp_path):
    from clipping import engine

    monkeypatch.chdir(tmp_path)
    for name in ("GOOGLE_API_KEY", "NVIDIA_API_KEY", "PEXELS_API_KEY", "HF_TOKEN"):
        monkeypatch.setenv(name, "")
    cfg = config.build_config(["--url", "https://example.test/live", "--live", "--upload-drive"])
    assert cfg.upload_drive is True
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=lambda *a, **k: None))
    monkeypatch.setattr(live, "build_window_video", lambda segs, out: out)
    monkeypatch.setattr(engine, "transcribe_video", lambda path, **k: ("[0 - 1] hi", []))

    def render(win_cfg, transcript, segs, clip_filter):
        kept = clip_filter([{"rank": 1, "start_time": 10, "end_time": 40, "viral_score": 95}])
        return [{"rank": c["rank"], "status": "success", "video_path": "v.mp4"} for c in kept]

    monkeypatch.setattr(runner, "analyze_and_render", render)
    uploaded = []

    class FakeUploader:
        root_folder = "Live Clips"

        @classmethod
        def from_env(cls):
            return cls()

        def run_folder(self, name):
            return "f"

        def upload_clip(self, row, run_name):
            uploaded.append((row["live_stream_start"], run_name))
            return "https://drive/x"

    monkeypatch.setattr(drive_upload, "DriveUploader", FakeUploader)

    class Rec:
        title, segments, ended = "My Stream", [], False

        def start(self): pass
        def stop(self): pass

        def poll(self):
            if self.recorded_seconds < 300:
                self.segments += [live.Segment("x.ts", self.recorded_seconds + i * 10, 10) for i in range(10)]
            else:
                self.ended = True

        @property
        def recorded_seconds(self):
            return self.segments[-1].end if self.segments else 0.0

    manifest = live.run_live(cfg, recorder=Rec(), sleep=lambda s: None)
    assert uploaded and uploaded[0][0] == 10 and uploaded[0][1].endswith("My Stream")
    assert manifest[0]["drive_link"] == "https://drive/x"


def test_live_continues_when_drive_setup_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GDRIVE_TOKEN_JSON", raising=False)
    cfg = config.build_config(["--url", "https://example.test/live", "--live", "--upload-drive"])
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=lambda *a, **k: None))

    class Rec:
        title, segments, ended = "", [], True
        recorded_seconds = 0.0

        def start(self): pass
        def stop(self): pass
        def poll(self): pass

    assert live.run_live(cfg, recorder=Rec(), sleep=lambda s: None) == []
    assert "Google Drive upload disabled" in capsys.readouterr().out
