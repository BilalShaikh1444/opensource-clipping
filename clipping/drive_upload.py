"""
clipping.drive_upload — copy finished clips to Google Drive as they are made.

Used by live mode (``--upload-drive``) so clips survive a Kaggle/Colab session
that is stopped or restarted. Authentication is a personal OAuth token with the
``drive.file`` scope (the app can only see files it created), created once with
``scripts/generate_drive_token.py`` and passed via the ``GDRIVE_TOKEN_JSON``
environment variable (a Kaggle secret). Service accounts cannot be used: they
have no storage quota in personal Google Drives.

Upload failures are reported and never stop clipping.
"""

from __future__ import annotations

import json
import os
import re

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
FOLDER_MIME = "application/vnd.google-apps.folder"
DEFAULT_ROOT_FOLDER = "Live Clips"


def _safe_name(text: str, limit: int = 80) -> str:
    text = re.sub(r"[\\/:*?\"<>|\r\n\t]+", " ", str(text or "")).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit].rstrip() or "clip"


def clip_basename(row: dict) -> str:
    """'0h23m10s - Title' so files sort by position in the stream."""
    s = int(row.get("live_stream_start") or 0)
    stamp = f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"
    return f"{stamp} - {_safe_name(row.get('youtube_title_final') or row.get('title_inggris'), 70)}"


def upload_info_text(row: dict) -> str:
    """Copy-paste helper for posting the clip manually."""
    tags = row.get("youtube_tags_final") or []
    s = int(row.get("live_stream_start") or 0)
    return (
        f"TITLE:\n{row.get('youtube_title_final') or row.get('title_inggris') or ''}\n\n"
        f"DESCRIPTION:\n{row.get('youtube_description_final') or ''}\n\n"
        f"TAGS:\n{', '.join(tags) if isinstance(tags, list) else tags}\n\n"
        f"TIKTOK CAPTION:\n{row.get('tiktok_caption_final') or ''}\n\n"
        f"SOURCE: {row.get('source_url') or ''} (at {s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d})\n"
    )


class DriveUploader:
    def __init__(self, service, root_folder: str = DEFAULT_ROOT_FOLDER, log=print):
        self.service = service
        self.root_folder = root_folder
        self.log = log
        self._folder_cache: dict = {}

    @classmethod
    def from_env(cls, env=os.environ, log=print) -> "DriveUploader":
        """Build from GDRIVE_TOKEN_JSON (authorized-user JSON). Raises with a clear message if unusable."""
        raw = (env.get("GDRIVE_TOKEN_JSON") or "").strip()
        if not raw:
            raise RuntimeError("GDRIVE_TOKEN_JSON is not set. Create it with scripts/generate_drive_token.py.")
        try:
            info = json.loads(raw)
        except json.JSONDecodeError:
            raise RuntimeError("GDRIVE_TOKEN_JSON is not valid JSON; paste the whole drive_token.json content.") from None
        if not info.get("refresh_token"):
            raise RuntimeError("GDRIVE_TOKEN_JSON has no refresh_token; rerun scripts/generate_drive_token.py.")
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        creds = Credentials.from_authorized_user_info(info, SCOPES)
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return cls(service, env.get("GDRIVE_FOLDER") or DEFAULT_ROOT_FOLDER, log=log)

    # ── folders ────────────────────────────────────────────────────────────
    def folder(self, name: str, parent: str | None = None) -> str:
        """Find (among folders this app created) or create a folder; returns its id."""
        key = (name, parent)
        if key in self._folder_cache:
            return self._folder_cache[key]
        escaped = name.replace("\\", "\\\\").replace("'", "\\'")
        q = f"name = '{escaped}' and mimeType = '{FOLDER_MIME}' and trashed = false"
        if parent:
            q += f" and '{parent}' in parents"
        found = self.service.files().list(q=q, fields="files(id)", pageSize=1).execute().get("files", [])
        if found:
            fid = found[0]["id"]
        else:
            body = {"name": name, "mimeType": FOLDER_MIME}
            if parent:
                body["parents"] = [parent]
            fid = self.service.files().create(body=body, fields="id").execute()["id"]
        self._folder_cache[key] = fid
        return fid

    def run_folder(self, run_name: str) -> str:
        return self.folder(_safe_name(run_name, 120), self.folder(self.root_folder))

    # ── files ──────────────────────────────────────────────────────────────
    def upload_file(self, path: str, folder_id: str, name: str | None = None, mime: str | None = None) -> dict:
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(path, mimetype=mime, resumable=True, chunksize=8 * 1024 * 1024)
        request = self.service.files().create(
            body={"name": name or os.path.basename(path), "parents": [folder_id]},
            media_body=media, fields="id, webViewLink",
        )
        response = None
        while response is None:
            _, response = request.next_chunk()
        return response

    def upload_text(self, text: str, folder_id: str, name: str) -> dict:
        from googleapiclient.http import MediaInMemoryUpload

        media = MediaInMemoryUpload(text.encode("utf-8"), mimetype="text/plain", resumable=False)
        return self.service.files().create(
            body={"name": name, "parents": [folder_id]}, media_body=media, fields="id, webViewLink",
        ).execute()

    def upload_clip(self, row: dict, run_name: str) -> str | None:
        """Upload video, thumbnail and an info .txt for one manifest row. Returns the video link or None."""
        video = row.get("video_path")
        if row.get("status") != "success" or not video or not os.path.isfile(video):
            return None
        try:
            folder_id = self.run_folder(run_name)
            base = clip_basename(row)
            res = self.upload_file(video, folder_id, f"{base}.mp4", "video/mp4")
            thumb = row.get("thumbnail_path")
            if thumb and os.path.isfile(thumb):
                self.upload_file(thumb, folder_id, f"{base}.jpg", "image/jpeg")
            self.upload_text(upload_info_text(row), folder_id, f"{base}.txt")
            link = res.get("webViewLink")
            self.log(f"   ☁️ Uploaded to Google Drive: {self.root_folder}/{_safe_name(run_name, 120)}/{base}.mp4")
            return link
        except Exception as e:  # never stop clipping because of Drive
            self.log(f"   ⚠️ Google Drive upload failed ({type(e).__name__}: {str(e)[:200]}). The clip is still in outputs/.")
            return None
