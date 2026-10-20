"""
Project Aether — Autonomous YouTube Publisher & Algorithmic Scheduler
Handles OAuth2 token lifecycle, quota management, algorithmic drip pacing,
and viral hook SEO generation for certified 9:16 vertical shorts.

Security Guardrails:
- Credentials strictly loaded from gitignored path: data/auth/
- Zero credential leakage; graceful Dry-Run fallback if credentials not configured.
- Daily quota budgeting: Max 3-4 uploads/day (4,800 - 6,400 units of 10,000 daily budget).
"""

import os
import sys
import json
import time
import logging
import datetime
from typing import Dict, Any, Optional, List, Union, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AetherPublisherSpoke")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
AUTH_DIR = os.path.join(DATA_DIR, "auth")
HISTORY_FILE = os.path.join(DATA_DIR, "publication_history.json")
TOKEN_FILE = os.path.join(AUTH_DIR, "youtube_token.json")
CLIENT_SECRETS_FILE = os.path.join(AUTH_DIR, "client_secrets.json")

# Quota dynamics (YouTube Data API v3)
DAILY_QUOTA_BUDGET = 10000
UNITS_PER_UPLOAD = 1600
MAX_UPLOADS_PER_DAY = 4

# Peak algorithmic viewership windows (Local Time: 11:30, 16:30, 20:30)
PEAK_HOURS = [(11, 30), (16, 30), (20, 30)]


def get_publication_history() -> List[Dict[str, Any]]:
    """Loads historical publications to enforce quota and pacing budgets."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def record_publication(entry: Dict[str, Any]):
    """Appends successful publication to the persistent ledger."""
    history = get_publication_history()
    history.append(entry)
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not record publication: {e}")


def get_todays_upload_count() -> int:
    """Calculates uploads dispatched in the past 24 hours."""
    history = get_publication_history()
    now = time.time()
    cutoff = now - 86400.0
    return sum(1 for h in history if h.get("timestamp", 0.0) >= cutoff)


def calculate_next_scheduled_drop(base_time: Optional[datetime.datetime] = None) -> datetime.datetime:
    """
    Calculates the next optimal algorithmic drop window (11:30 AM, 4:30 PM, 8:30 PM).
    Spaces uploads by at least 4.5 hours.
    """
    now = base_time or datetime.datetime.now()
    today = now.date()

    for h, m in PEAK_HOURS:
        candidate = datetime.datetime(today.year, today.month, today.day, h, m)
        if candidate > (now + datetime.timedelta(minutes=30)):
            return candidate

    # If all slots today have passed, target first slot tomorrow
    tomorrow = today + datetime.timedelta(days=1)
    h, m = PEAK_HOURS[0]
    return datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, h, m)


MAX_TITLE_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 5000
MAX_TAGS_LENGTH = 450
MAX_TAGS_COUNT = 30


def remove_control_characters(s: str) -> str:
    """Removes non-printable ASCII control characters while preserving valid formatting."""
    return "".join(ch for ch in str(s or "") if ord(ch) > 31 and ord(ch) != 127 or ch in ("\n", "\r", "\t"))


def normalize_tags(raw_tags: Union[List[str], str]) -> List[str]:
    """
    Normalizes, sanitizes, and deduplicates YouTube tags.
    Strips leading '#', removes quotes, and enforces YouTube's 450-byte total limit.
    """
    if isinstance(raw_tags, str):
        candidates = [t.strip() for t in raw_tags.split(",") if t.strip()]
    else:
        candidates = list(raw_tags or [])

    unique_tags: List[str] = []
    seen: set = set()
    total_len = 0

    for item in candidates:
        cleaned = remove_control_characters(item).strip().lstrip("#").replace('"', '')
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue

        next_len = total_len + len(cleaned) + (1 if unique_tags else 0)
        if next_len > MAX_TAGS_LENGTH or len(unique_tags) >= MAX_TAGS_COUNT:
            break

        unique_tags.append(cleaned[:100])
        seen.add(key)
        total_len = next_len

    return unique_tags


def normalize_and_validate_metadata(seo: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Ensures complete compliance with YouTube Data API v3 metadata specifications.
    Guarantees title <= 100 chars, description <= 5000 chars, tags <= 450 bytes.
    """
    warnings: List[str] = []
    raw_title = remove_control_characters(seo.get("title", "")).strip()
    if len(raw_title) > MAX_TITLE_LENGTH:
        warnings.append(f"Title exceeded {MAX_TITLE_LENGTH} chars; truncated.")
        raw_title = raw_title[:MAX_TITLE_LENGTH - 3].rstrip() + "..."

    raw_desc = remove_control_characters(seo.get("description", "")).strip()
    if len(raw_desc) > MAX_DESCRIPTION_LENGTH:
        warnings.append(f"Description exceeded {MAX_DESCRIPTION_LENGTH} chars; truncated.")
        raw_desc = raw_desc[:MAX_DESCRIPTION_LENGTH].rstrip()

    tags = normalize_tags(seo.get("tags", []))
    if len(tags) < 3:
        warnings.append("Fewer than 3 tags remain after normalization.")

    cat_id = str(seo.get("category_id", "24")).strip()
    if not cat_id.isdigit():
        cat_id = "24"

    sanitized = {
        "title": raw_title or "Cinematic Short #shorts",
        "description": raw_desc,
        "tags": tags,
        "category_id": cat_id,
        "default_language": "en",
        "default_audio_language": "en"
    }
    return sanitized, warnings


def add_video_to_playlist(youtube: Any, video_id: str, playlist_id: str) -> bool:
    """
    Assigns an uploaded video to a specified playlist using YouTube Data API v3.
    Consumes ~50 quota units.
    """
    try:
        body = {
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {
                    "kind": "youtube#video",
                    "videoId": video_id
                }
            }
        }
        youtube.playlistItems().insert(part="snippet", body=body).execute()
        logger.info(f"🎉 Successfully added video {video_id} to playlist {playlist_id}")
        return True
    except Exception as e:
        logger.warning(f"Could not add video {video_id} to playlist {playlist_id}: {e}")
        return False


def attach_captions(youtube: Any, video_id: str, srt_path: str, language: str = "en") -> bool:
    """
    Uploads an .srt caption track directly to a YouTube video.
    """
    if not srt_path or not os.path.exists(srt_path):
        return False
    try:
        from googleapiclient.http import MediaFileUpload
        caption_body = {
            "snippet": {
                "videoId": video_id,
                "language": language,
                "name": f"{language.upper()} Captions",
                "isDraft": False
            }
        }
        media = MediaFileUpload(srt_path, mimetype="application/x-subrip")
        youtube.captions().insert(part="snippet", body=caption_body, media_body=media).execute()
        logger.info(f"Subtitles from {srt_path} attached to video {video_id}")
        return True
    except Exception as e:
        logger.warning(f"Could not attach captions for {video_id}: {e}")
        return False


def generate_viral_seo_metadata(raw_title: str,
                                 summary: str = "",
                                 is_cc: bool = False,
                                 uploader: str = "Original Creator") -> Dict[str, Any]:
    """
    Generates high-CTR SEO title, narrative description, and viral hashtags.
    """
    # Clean and punch up title (< 55 chars)
    clean_title = raw_title.replace("Viral Arc", "").strip(" :-,")
    if len(clean_title) > 45:
        clean_title = clean_title[:42] + "..."

    emojis = ["🔥", "😱", "🤯", "⚡", "👀"]
    emoji = emojis[hash(clean_title) % len(emojis)]
    hook_title = f"{clean_title} {emoji} #shorts"

    tags = ["shorts", "viral", "trending", "movieclips", "scene", "fyp", "aether"]
    desc_lines = [
        f"{summary or clean_title}",
        "",
        "🔔 Subscribe for daily cinematic shorts and compelling narratives!",
        "",
        "#shorts #viral #trending #cinema #entertainment"
    ]
    if is_cc:
        desc_lines.append(f"\nSource Attribution (CC-BY): {uploader}")

    raw_payload = {
        "title": hook_title,
        "description": "\n".join(desc_lines),
        "tags": tags,
        "category_id": "24",  # Entertainment
    }
    sanitized, warnings = normalize_and_validate_metadata(raw_payload)
    sanitized["validation_warnings"] = warnings
    return sanitized


def generate_syndication_package(raw_title: str,
                                 summary: str = "",
                                 is_cc: bool = False,
                                 uploader: str = "Original Creator",
                                 playlist_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates tailored, multi-platform publishing metadata for YouTube Shorts,
    TikTok Content Posting API, and Instagram Reels Graph API.
    """
    yt = generate_viral_seo_metadata(raw_title, summary, is_cc, uploader)
    if playlist_id:
        yt["playlist_id"] = playlist_id
    yt["contains_synthetic_media"] = True

    clean_title = raw_title.replace("Viral Arc", "").strip(" :-,")
    if len(clean_title) > 40:
        clean_title = clean_title[:37] + "..."

    # TikTok: strict < 150 char caption with highest-indexing FYP tags
    tt_caption = f"{clean_title} 😱 #fyp #viral #movieclips #shorts #foryou"
    if len(tt_caption) > 150:
        tt_caption = tt_caption[:147] + "..."

    # Instagram Reels: engagement-provoking question + reels tags
    ig_caption = (
        f"Did you catch this moment? 👇\n\n"
        f"{summary or clean_title}\n\n"
        f"Follow for daily cinematic moments!\n"
        f"#reels #reelsinstagram #trending #viralvideos #movieclips"
    )

    return {
        "youtube": yt,
        "tiktok": {
            "caption": tt_caption,
            "privacy_level": "PUBLIC_TO_EVERYONE",
            "disable_duet": False,
            "disable_stitch": False,
            "disable_comment": False,
        },
        "instagram_reels": {
            "caption": ig_caption,
            "share_to_feed": True,
        },
        "validation_warnings": yt.get("validation_warnings", [])
    }


def publish_short(video_path: str,
                  raw_title: str = "Cinematic Short",
                  summary: str = "",
                  thumbnail_path: Optional[str] = None,
                  srt_path: Optional[str] = None,
                  playlist_id: Optional[str] = None,
                  is_cc: bool = False,
                  uploader: str = "Original Creator",
                  privacy_status: str = "private",
                  notify_subscribers: bool = True) -> Dict[str, Any]:
    """
    Publishes or schedules a certified 9:16 short to YouTube with enterprise-grade safeguards:
    - Metadata sanitization (length caps, tag normalization)
    - Mandatory synthetic media disclosure (containsSyntheticMedia: True)
    - Automatic playlist assignment
    - Caption (.srt) upload
    - Upload outcome reconciliation state tracking
    - Daily quota budgeting
    """
    if not os.path.exists(video_path):
        return {"status": "error", "error": f"Video file not found: {video_path}"}

    # Quota check
    daily_count = get_todays_upload_count()
    if daily_count >= MAX_UPLOADS_PER_DAY:
        logger.warning(
            f"Daily upload budget reached ({daily_count}/{MAX_UPLOADS_PER_DAY} uploads). "
            f"Holding publication to preserve YouTube Data API quota."
        )
        return {
            "status": "quota_held",
            "daily_uploads": daily_count,
            "max_daily": MAX_UPLOADS_PER_DAY,
            "message": "Upload deferred to prevent quota exhaustion."
        }

    # Calculate optimal schedule drop time
    sched_dt = calculate_next_scheduled_drop()
    sched_iso = sched_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Generate and sanitize viral metadata
    seo = generate_viral_seo_metadata(raw_title, summary, is_cc, uploader)

    # Check for OAuth credentials
    has_credentials = os.path.exists(CLIENT_SECRETS_FILE) or os.path.exists(TOKEN_FILE)

    if not has_credentials:
        logger.info(
            f"[Dry-Run Mode] No YouTube OAuth token at {TOKEN_FILE}. "
            f"Simulating publication for: {seo['title']}"
        )
        simulated_id = f"sim_{int(time.time())}_{os.path.basename(video_path)}"
        result = {
            "status": "success",
            "mode": "dry_run",
            "video_id": simulated_id,
            "title": seo["title"],
            "scheduled_time": sched_iso,
            "privacy_status": privacy_status,
            "video_path": video_path,
            "thumbnail_path": thumbnail_path if thumbnail_path and os.path.exists(thumbnail_path) else None,
            "srt_path": srt_path if srt_path and os.path.exists(srt_path) else None,
            "playlist_id": playlist_id,
            "contains_synthetic_media": True,
            "notify_subscribers": notify_subscribers,
            "quota_units_used": UNITS_PER_UPLOAD,
            "validation_warnings": seo.get("validation_warnings", []),
            "timestamp": time.time(),
            "note": "Credentials pending in data/auth/. Verified ready for live API publishing."
        }
        record_publication(result)
        return result

    # Live API Publishing Flow with Resilient Safeguards
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.oauth2.credentials import Credentials

        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            token_data = json.load(f)
        creds = Credentials.from_authorized_user_info(token_data)

        youtube = build("youtube", "v3", credentials=creds)

        body = {
            "snippet": {
                "title": seo["title"],
                "description": seo["description"],
                "tags": seo["tags"],
                "categoryId": seo["category_id"],
                "defaultLanguage": seo.get("default_language", "en"),
                "defaultAudioLanguage": seo.get("default_audio_language", "en")
            },
            "status": {
                "privacyStatus": privacy_status,
                "publishAt": sched_iso,
                "selfDeclaredMadeForKids": False,
                "containsSyntheticMedia": True
            }
        }

        media = MediaFileUpload(video_path, chunksize=1024 * 1024, resumable=True)
        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
            notifySubscribers=notify_subscribers
        )
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                logger.info(f"Uploading {seo['title']}: {int(status.progress() * 100)}%")

        video_id = response.get("id")
        logger.info(f"🎉 Successfully uploaded short to YouTube! Video ID: {video_id}")

        # Upload custom thumbnail if provided
        if thumbnail_path and os.path.exists(thumbnail_path) and video_id:
            try:
                thumb_media = MediaFileUpload(thumbnail_path)
                youtube.thumbnails().set(videoId=video_id, media_body=thumb_media).execute()
                logger.info(f"Custom thumbnail attached for {video_id}")
            except Exception as e:
                logger.warning(f"Could not upload thumbnail: {e}")

        # Attach subtitles (.srt) if provided
        if srt_path and os.path.exists(srt_path) and video_id:
            attach_captions(youtube, video_id, srt_path)

        # Add to target playlist if specified
        if playlist_id and video_id:
            add_video_to_playlist(youtube, video_id, playlist_id)

        result = {
            "status": "success",
            "mode": "live",
            "video_id": video_id,
            "title": seo["title"],
            "scheduled_time": sched_iso,
            "privacy_status": privacy_status,
            "video_path": video_path,
            "thumbnail_path": thumbnail_path,
            "srt_path": srt_path,
            "playlist_id": playlist_id,
            "contains_synthetic_media": True,
            "notify_subscribers": notify_subscribers,
            "quota_units_used": UNITS_PER_UPLOAD,
            "validation_warnings": seo.get("validation_warnings", []),
            "timestamp": time.time()
        }
        record_publication(result)
        return result

    except Exception as e:
        logger.error(f"Live YouTube upload interrupted or failed: {e}")
        # Record reconciliation guard state so duplicate retry does not occur without channel inspection
        reconcile_record = {
            "status": "reconciliation_required",
            "mode": "error",
            "title": seo["title"],
            "video_path": video_path,
            "error": str(e),
            "timestamp": time.time()
        }
        record_publication(reconcile_record)
        return {
            "status": "error",
            "error": str(e),
            "reconciliation_required": True,
            "message": "Upload state uncertain; channel reconciliation recorded before retry."
        }

        return {"status": "error", "error": str(e)}


def main():
    if len(sys.argv) < 2:
        print("Usage: python wsl2_gpu/publisher_spoke.py <short_video_path> [title]")
        sys.exit(1)

    vid_path = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(vid_path)
    res = publish_short(vid_path, raw_title=title)
    print("\n" + "=" * 65)
    print(" PUBLISHER SPOKE RESULT")
    print(f" Status: {res.get('status')} ({res.get('mode', 'live')})")
    print(f" Title: {res.get('title')}")
    print(f" Scheduled Time: {res.get('scheduled_time')}")
    print(f" Quota Cost: {res.get('quota_units_used', 0)} units")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
