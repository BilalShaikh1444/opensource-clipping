#!/usr/bin/env python3
"""
generate_drive_token.py — one-time Google Drive sign-in for live clip uploads.

Run this ON YOUR OWN COMPUTER (it opens a browser):

    python scripts/generate_drive_token.py

Before running: download your Google Cloud "Desktop app" OAuth client JSON and
save it as .credentials/drive_client_secret.json (see README, "Upload clips to
Google Drive").

It writes .credentials/drive_token.json. Put the whole content of that file in
a Kaggle secret named GDRIVE_TOKEN_JSON. The token only allows access to files
this app creates (drive.file scope). Treat it like a password; .credentials/ is
git-ignored. To revoke it later: https://myaccount.google.com/permissions
"""

import json
import os
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
CLIENT_SECRET_FILE = os.path.join(".credentials", "drive_client_secret.json")
TOKEN_FILE = os.path.join(".credentials", "drive_token.json")


def main() -> int:
    if not os.path.isfile(CLIENT_SECRET_FILE):
        print(f"Missing {CLIENT_SECRET_FILE}.\n"
              "Google Cloud Console → APIs & Services → Credentials → Create credentials → "
              "OAuth client ID → Desktop app → Download JSON, then save it there.")
        return 1
    os.makedirs(".credentials", exist_ok=True)
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, scopes=SCOPES)
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    if not creds.refresh_token:
        print("Google did not return a refresh token. Remove this app at "
              "https://myaccount.google.com/permissions and run again.")
        return 1
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(creds.to_json())

    # Quick check: create (or find) the clips folder.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from clipping.drive_upload import DriveUploader
    uploader = DriveUploader.from_env({"GDRIVE_TOKEN_JSON": json.dumps(json.loads(creds.to_json()))})
    uploader.folder(uploader.root_folder)
    print(f"\nSaved {TOKEN_FILE} and checked access: folder '{uploader.root_folder}' is ready in your Drive.")
    print("Next: copy the whole content of that file into a Kaggle secret named GDRIVE_TOKEN_JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
