"""--content-profile: gaming prompt, category set, metadata checks, CLI."""

import json
import re
import types

import pytest

from clipping import config, engine, metadata
from clipping.gaming_prompt import GAMING_TARGET_ACCOUNTS


def _cfg(**kw):
    base = dict(hook_v2=False, hook_v2_items=3, hook_v2_style="glitch_fast",
                no_segment_trim=False, silence_trim=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_podcast_profile_is_the_unchanged_default_prompt():
    legacy = engine.get_analysis_prompt("[0.0 - 1.0] halo", 5, 3, cfg=_cfg())
    explicit = engine.get_analysis_prompt("[0.0 - 1.0] halo", 5, 3, cfg=_cfg(content_profile="podcast"))
    assert legacy == explicit
    assert "Mereska" in legacy and "Bahasa Indonesia" in legacy
    assert engine.get_target_accounts(_cfg()) is engine.TARGET_ACCOUNTS
    assert engine.get_target_accounts(None) is engine.TARGET_ACCOUNTS


def test_gaming_prompt_content():
    p = engine.get_analysis_prompt("[12.0 - 14.5] NO WAY bro", 4, 3, cfg=_cfg(content_profile="gaming"))
    assert p.rstrip().endswith("[12.0 - 14.5] NO WAY bro")
    assert "Find the 4 strongest moments" in p
    assert f"Duration {engine.MIN_CLIP_DURATION}-{engine.MAX_CLIP_DURATION} seconds" in p
    assert "Mereska" not in p and "Bahasa" not in p
    for tipe, acc in GAMING_TARGET_ACCOUNTS.items():
        assert tipe in p and acc["akun_tujuan"] in p
    assert "{{" not in p and "}}" not in p  # f-string braces rendered
    assert "SEGMENT TRIMMING" in p and "HOOK V2" not in p


def test_gaming_prompt_json_example_is_valid_and_complete():
    p = engine.get_analysis_prompt("T", 1, 3, cfg=_cfg(content_profile="gaming"))
    block = p.split("(use these exact field names):", 1)[1].split("Transcript:", 1)[0]
    example = json.loads(block)
    required = {
        "rank", "viral_score", "hook_start_time", "hook_end_time", "start_time", "end_time",
        "typography_plan", "broll_list", "recommended_visual_broll_hook", "alasan", "bgm_mood",
        "title_indonesia", "title_inggris", "hastag", "description_hook", "description_context",
        "keyword_tags", "tiktok_title_id", "tiktok_caption_id", "tiktok_caption",
        "klasifikasi_akun", "hook_v2", "keep_segments",
    }
    assert required <= set(example[0])
    assert example[0]["klasifikasi_akun"]["tipe_akun"] in GAMING_TARGET_ACCOUNTS


def test_gaming_prompt_optional_sections():
    p = engine.get_analysis_prompt(
        "T", 1, 3,
        cfg=_cfg(content_profile="gaming", hook_v2=True, hook_v2_items=4, no_segment_trim=True),
    )
    assert "4 very short cuts" in p and "SEGMENT TRIMMING" not in p
    p = engine.get_analysis_prompt("T", 1, 3, cfg=_cfg(content_profile="gaming", silence_trim=True))
    assert "AGGRESSIVELY cut silence" in p


def test_gaming_profile_uses_gaming_categories():
    assert engine.get_target_accounts(_cfg(content_profile="gaming")) is GAMING_TARGET_ACCOUNTS


def test_english_gaming_metadata_passes_normalization():
    item = {
        "rank": 1, "viral_score": 90, "start_time": 1, "end_time": 30,
        "title_indonesia": "Heist goes wrong", "title_inggris": "The heist went wrong in seconds",
        "hastag": "#gta6 #gaming", "description_hook": "One wrong turn ruined everything.",
        "description_context": "A GTA 6 stream heist turns into a chaotic police chase.",
        "keyword_tags": ["gta 6", "heist fail", "police chase", "funny moments", "gta 6 stream"],
        "tiktok_title_id": "GTA 6 heist turns into the wildest police chase",
        "tiktok_caption_id": "The getaway lasted about four seconds.",
        "tiktok_caption": "Getaway driver of the year.",
        "klasifikasi_akun": {"tipe_akun": "Funny", "akun_tujuan": "Gaming.Funny"},
    }
    out = metadata.normalize_and_validate([item])
    assert out[0]["youtube_title_final"] == "The heist went wrong in seconds"
    assert out[0]["tiktok_title_id_final"] == "GTA 6 heist turns into the wildest police chase"
    assert out[0]["tiktok_caption_final"].endswith("#gta6 #gaming")


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("GOOGLE_API_KEY", "NVIDIA_API_KEY", "PEXELS_API_KEY", "HF_TOKEN"):
        monkeypatch.setenv(name, "")


def test_cli_content_profile(clean_env):
    assert config.build_config(["--url", "https://x.test"]).content_profile == "podcast"
    assert config.build_config(["--url", "https://x.test", "--content-profile", "gaming"]).content_profile == "gaming"
    with pytest.raises(SystemExit):
        config.build_config(["--url", "https://x.test", "--content-profile", "cooking"])
