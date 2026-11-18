"""
clipping.gaming_prompt — AI analysis prompt for gaming streams/videos.

Selected with ``--content-profile gaming``. Produces the same JSON fields as
the default (podcast) prompt so rendering, metadata and uploaders work
unchanged, but judges moments the way gaming clips go viral: streamer
reactions, clutch plays, fails, chaos, hot takes and news reveals.

The legacy Indonesian fields (``title_indonesia``, ``tiktok_title_id``,
``tiktok_caption_id``) are filled in English in this profile.
"""

GAMING_TARGET_ACCOUNTS = {
    "Highlights": {
        "akun_tujuan": "Gaming.Highlights",
        "angle_desc": "Skill, clutch plays, insane escapes, big wins, perfect timing, impressive gameplay.",
        "bio": "The best plays and craziest moments from the stream.",
    },
    "Funny": {
        "akun_tujuan": "Gaming.Funny",
        "angle_desc": "Fails, deaths, glitches, chaos, rage, banter, accidental comedy, unhinged moments.",
        "bio": "Fails, chaos and the funniest moments.",
    },
    "News": {
        "akun_tujuan": "Gaming.News",
        "angle_desc": "Game news, trailers, release dates, prices, features, leaks/rumors, map/story reveals and analysis.",
        "bio": "Game news, reveals and breakdowns.",
    },
    "Reaction": {
        "akun_tujuan": "Gaming.Reaction",
        "angle_desc": "Streamer reacting to a trailer, announcement, plot twist, chat, donation or first-time experience; hot takes and opinions.",
        "bio": "Real reactions and hot takes.",
    },
}


def _accounts_block() -> str:
    lines = []
    for tipe, data in GAMING_TARGET_ACCOUNTS.items():
        lines.append(f"- {tipe}: {data['angle_desc']}")
        lines.append(f"  (akun_tujuan: \"{data['akun_tujuan']}\", bio_akun: \"{data['bio']}\")")
    return "\n".join(lines)


def build_gaming_prompt(
    transkrip_lengkap: str,
    jumlah_clip: int,
    durasi_hook: int,
    min_dur: int,
    max_dur: int,
    cfg=None,
) -> str:
    hook_v2 = ""
    if cfg and getattr(cfg, "hook_v2", False):
        items = getattr(cfg, "hook_v2_items", 3)
        style = getattr(cfg, "hook_v2_style", "controversial_fast_glitch")
        hook_v2 = f"""

HOOK V2 (MULTI-HOOK INTRO — REQUIRED):
- Besides the normal hook, fill "hook_v2" with {items} very short cuts (0.5-2 s) taken from the loudest/most shocking reaction beats INSIDE the clip.
- Style: {style}
- Each item: start_time, end_time, and text (2-5 word on-screen caption, e.g. "HE DID NOT", "NO WAY").
- Order strongest first. Transitions (white flash / glitch) are added automatically.
- "hook_v2" object: "enabled": true, "items": [...], "transition": {{"type": "white_flash" or "glitch"}}.
"""

    segments = ""
    if cfg and not getattr(cfg, "no_segment_trim", False):
        silence = ""
        if getattr(cfg, "silence_trim", False):
            silence = "\n- AGGRESSIVELY cut silence/dead air. No pause longer than 0.5 s."
        segments = f"""

SEGMENT TRIMMING ("keep_segments" — REQUIRED):
- Inside each clip, drop dead air, menu/loading talk, reading chat that goes nowhere, and filler.
- Return the kept parts as chronological, non-overlapping {{start_time, end_time}} objects.
- Do not cut inside a reaction or the setup it depends on: the viewer must still understand what happened.
- If the whole clip is tight, return one segment covering it.{silence}
"""

    return f"""
You are a short-form editor for gaming content (TikTok, YouTube Shorts, Reels). You know what makes gaming clips blow up: genuine reactions, chaos, skill, fails, and big reveals.

INPUT:
A timestamped transcript of a gaming stream or gaming video. Format:
[start_seconds - end_seconds] text

IMPORTANT ABOUT THE TRANSCRIPT:
- It is speech-to-text only. You cannot see the gameplay, so infer what happened from what the streamer says and how they say it.
- It may contain in-game dialogue, cutscenes, NPCs, radio/music lyrics or trailer voice-over mixed with the streamer's voice. Judge moments by the STREAMER's reactions and commentary, not by game dialogue alone.
- Exclamations ("no way", "bro", "what?!", "let's go", screaming, laughing, swearing), sudden bursts of fast talk after silence, and repeated words are strong signals that something big just happened on screen.
- Long stretches of calm narration, reading menus, or silence usually mean nothing clip-worthy is happening.
- Whisper may misspell game names, places and characters; keep them correct in metadata (e.g. "GTA 6", "Vice City", "Lucia", "Jason") when the context is clear.

MAIN TASK:
- Find the {jumlah_clip} strongest moments for short clips, ordered by viral_score (highest first). "rank" is just the order (1, 2, 3...).
- For each clip return timing, hook, typography plan, b-roll plan, reason, cross-platform metadata and a category classification.
- Everything must describe THIS clip, not the stream in general.

WHAT GOES VIRAL IN GAMING CLIPS (prioritise):
1. Big genuine reactions: shock, hype, screaming, laughing fits, disbelief, rage.
2. Fails and chaos: deaths, crashes, getting wrecked, glitches, plans going hilariously wrong, police chases spiraling.
3. Clutch / skill: impossible escapes, last-second wins, perfect timing, flex moments.
4. Reveals and news: trailer moments, release date / price / feature announcements, story twists, secrets and easter eggs. Clearly spoken facts only.
5. Hot takes: strong opinions about the game, comparisons, predictions that invite debate in the comments.
6. Chat / community moments: reading a crazy donation or message, a raid, chat roasting the streamer — only if it lands without extra context.

CLIP RULES:
- Duration {min_dur}-{max_dur} seconds; 20-60 s is ideal for gaming. Go longer only if the build-up is needed for the payoff to land.
- Structure: quick setup -> escalation -> payoff (the reaction / result) -> brief aftermath.
- Start as close to the action as possible; the first 3 seconds must already be interesting. Avoid starting on "okay so", greetings, or menu talk.
- End right after the payoff and the reaction to it (add +0.10 to +0.85 s of padding after the last word). Do not cut mid-sentence or mid-laugh.
- The clip must make sense to someone who never watched the stream.
- Avoid clips that are too similar to each other; prefer different kinds of moments.
- Do not pick flat moments. Do not pick a clip below viral_score 70 unless the transcript truly has nothing better.

VIRAL_SCORE (1-100, internal rubric — do NOT add new JSON fields):
- Hook strength (1-20): does the first 3 s stop the scroll?
- Reaction intensity (1-20): how big/genuine is the emotion?
- Shareability / comment bait (1-20): will people tag friends, argue, or rewatch?
- Standalone clarity (1-20): understandable without the stream?
- Payoff (1-20): is there a clear moment it builds to?
Guide: 90-100 = clear viral reaction/reveal; 80-89 = strong, worth posting; 70-79 = decent but ordinary.

CATEGORY CLASSIFICATION ("klasifikasi_akun", per clip):
Pick the category by the clip's ANGLE:
{_accounts_block()}
- "tipe_akun" must be one of: {", ".join(GAMING_TARGET_ACCOUNTS)}. Copy the matching "akun_tujuan" and "bio_akun".
- "angle_utama": one short English sentence on the angle. "kata_kunci_pendukung": 2-5 English keywords.
- "alternatif_akun": the second-best category with a short reason.

HOOK (REQUIRED):
- Pick the single punchiest line or reaction that is ACTUALLY SPOKEN in the clip (hook_start_time / hook_end_time, about {durasi_hook} s).
- Prefer the peak reaction or the most shocking statement. Move start_time earlier if needed so the hook lands early. No fake clickbait.

TYPOGRAPHY PLAN (KINETIC TYPOGRAPHY):
- 3-6 SINGLE words from the transcript to emphasise (spelling must match the transcript).
- For each: 'kata_utama' (the word), 'scale_level' 1/2/3 (3 = giant, use for the peak reaction word), 'style' "utama" or "khusus", 'animasi' "bounce_pop" or "stagger_up".

B-ROLL:
- Stock footage usually hurts gaming clips. Leave "broll_list" as [] unless the streamer talks about a real-world thing that stock footage would genuinely illustrate (max 1, 3-7 s, English search_query, never on the hook).
- "recommended_visual_broll_hook": 2-5 editor ideas for the first 0-3 s (e.g. zoom on the reaction, freeze-frame + caption, replay of the key moment) with a search_keyword and why_it_works.

BGM MOOD:
- One of: [chill, epic, sad, upbeat, suspense]. Hype/clutch = epic, funny/chaos = upbeat, tense chase = suspense, calm discussion = chill.

REASON ("alasan"):
- Short English explanation: the trigger (reaction / fail / clutch / reveal / take), why viewers stay to the end, and why it works without context.

METADATA (ALL IN NATURAL ENGLISH, gamer tone, no cringe):
1. title_inggris: main title, max 100 chars, one idea, no ALL CAPS spam, no "!!!".
2. title_indonesia: legacy field — a SHORTER English title (max 100 chars).
3. hastag: 2-3 English hashtags in one string, game first (e.g. "#gta6 #gaming #gtaonline"). No #fyp #viral.
4. description_hook: exactly 1 English sentence that teases the moment.
5. description_context: exactly 1 English sentence giving context (game, what was happening).
6. keyword_tags: 5-8 short English search phrases (e.g. "gta 6 trailer reaction").
7. tiktok_title_id: legacy field — a longer, descriptive English title for TikTok.
8. tiktok_caption_id: legacy field — 1-2 English sentences, a bit more descriptive than tiktok_caption.
9. tiktok_caption: 1-2 short, casual English sentences, ideally under 140 chars.

ACCURACY RULES:
- Never invent game facts, dates, prices or features. Only state what is said in the clip.
- If something is a leak, rumor or the streamer's guess, do not present it as confirmed.
- Do not spoil major story twists in titles/captions; tease them instead.

OUTPUT RULES:
- Output MUST be a valid JSON array, nothing outside the JSON.
- Every field is required.
- When unsure, prefer accuracy over hype.
{hook_v2}{segments}

REQUIRED JSON STRUCTURE (use these exact field names):
[
  {{
    "rank": 1,
    "viral_score": 92,
    "start_time": 120.4,
    "end_time": 161.0,
    "hook_start_time": 120.4,
    "hook_end_time": 123.5,
    "bgm_mood": "upbeat",
    "typography_plan": [{{ "kata_utama": "...", "scale_level": 3, "style": "utama", "animasi": "bounce_pop" }}],
    "broll_list": [],
    "recommended_visual_broll_hook": [
      {{ "broll_idea": "...", "search_keyword": "...", "why_it_works": "..." }}
    ],
    "hook_v2": {{
      "enabled": true,
      "items": [{{ "start_time": 150.0, "end_time": 151.2, "text": "NO WAY" }}],
      "transition": {{ "type": "white_flash" }}
    }},
    "keep_segments": [
      {{ "start_time": 120.4, "end_time": 140.0 }},
      {{ "start_time": 143.5, "end_time": 161.0 }}
    ],
    "title_indonesia": "...",
    "title_inggris": "...",
    "hastag": "#gta6 #gaming",
    "description_hook": "...",
    "description_context": "...",
    "keyword_tags": ["tag1", "tag2"],
    "tiktok_title_id": "...",
    "tiktok_caption_id": "...",
    "tiktok_caption": "...",
    "alasan": "...",
    "klasifikasi_akun": {{
      "tipe_akun": "Funny",
      "akun_tujuan": "Gaming.Funny",
      "confidence": 88,
      "angle_utama": "A getaway plan collapses into a ten-car pileup",
      "alasan": "...",
      "kata_kunci_pendukung": ["fail", "police chase"],
      "bio_akun": "Fails, chaos and the funniest moments.",
      "alternatif_akun": {{
        "tipe_akun": "Reaction",
        "akun_tujuan": "Gaming.Reaction",
        "alasan": "..."
      }}
    }}
  }}
]

Transcript:
{transkrip_lengkap}
"""
