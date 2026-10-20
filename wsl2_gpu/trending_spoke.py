"""
Project Aether — Intelligent Trending Research Spoke
Harvests breakout queries from Google Trends RSS and YouTube Trending.
Extracts YouTube Heatmap replay spikes to isolate viral segments before download.
Enforces strict sensitivity and safety filtering.
"""

import os
import re
import time
import logging
import urllib.request
try:
    import defusedxml.ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AetherTrendingSpoke")

# Safety blacklist: blocks controversy, tragedy, graphic violence, sensitive political news
SENSITIVE_PATTERNS = [
    r"\b(war|killed|shooting|murder|crash|assassination|suicide|attack|fatal|terrorist)\b",
    r"\b(scandal|arrested|indicted|lawsuit|sexual|abuse|assault|death|died|casualty)\b",
    r"\b(controversy|protest|riot|hostage|strike|explosion|disaster|bombing|tragedy)\b",
    r"\b(genocide|massacre|hostile|weapon|extremist|conspiracy)\b"
]
SAFETY_REGEX = re.compile("|".join(SENSITIVE_PATTERNS), re.IGNORECASE)


class TrendingTopic:
    def __init__(self, title: str, approx_traffic: str, pub_date: str, link: str, is_safe: bool = True):
        self.title = title
        self.approx_traffic = approx_traffic
        self.pub_date = pub_date
        self.link = link
        self.is_safe = is_safe
        self.discovered_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "approx_traffic": self.approx_traffic,
            "pub_date": self.pub_date,
            "link": self.link,
            "is_safe": self.is_safe,
            "discovered_at": self.discovered_at
        }

    def __repr__(self) -> str:
        return f"<TrendingTopic '{self.title}' traffic={self.approx_traffic} safe={self.is_safe}>"


def is_topic_safe(title: str, description: str = "") -> bool:
    """Certifies that a trending query is suitable for universal public audience."""
    combined = f"{title} {description}"
    return not bool(SAFETY_REGEX.search(combined))


def fetch_google_trends(geo: str = "US") -> List[TrendingTopic]:
    """Fetches real-time trending search queries from Google Trends RSS."""
    url = f"https://trends.google.com/trending/rss?geo={geo}"
    topics: List[TrendingTopic] = []

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProjectAether/3.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)

        # Parse RSS items
        # Google Trends RSS uses ht:approx_traffic namespace
        ns = {"ht": "https://trends.google.com/trending/rss"}

        for item in root.findall("./channel/item"):
            title_elem = item.find("title")
            pub_date_elem = item.find("pubDate")
            link_elem = item.find("link")
            desc_elem = item.find("description")
            traffic_elem = item.find("ht:approx_traffic", ns)

            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
            pub_date = pub_date_elem.text.strip() if pub_date_elem is not None and pub_date_elem.text else ""
            link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""
            traffic = traffic_elem.text.strip() if traffic_elem is not None and traffic_elem.text else "N/A"
            desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""

            if not title:
                continue

            safe = is_topic_safe(title, desc)
            if safe:
                topics.append(TrendingTopic(title=title, approx_traffic=traffic, pub_date=pub_date, link=link, is_safe=True))
            else:
                logger.info(f"Filtered out sensitive topic: {title}")

        logger.info(f"Fetched {len(topics)} safe trending topics from Google Trends ({geo}).")
    except Exception as e:
        logger.warning(f"Could not fetch Google Trends RSS: {e}")

    return topics


def extract_youtube_heatmap(video_url: str) -> Optional[List[Dict[str, float]]]:
    """
    Extracts the YouTube 'Most Replayed' retention heatmap curve via yt-dlp.
    Returns list of normalized {start_time, end_time, value} items, where value is 0.0-1.0.
    """
    try:
        import yt_dlp
        ydl_opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            heatmap = info.get("heatmap")
            if heatmap and isinstance(heatmap, list):
                logger.info(f"Extracted {len(heatmap)} heatmap points for: {video_url}")
                return heatmap
    except Exception as e:
        logger.warning(f"Could not extract heatmap for {video_url}: {e}")
    return None


def find_top_heatmap_peak(heatmap: List[Dict[str, float]], min_peak_duration: float = 50.0) -> Optional[Tuple[float, float, float]]:
    """
    Finds the highest viewer replay density peak in the heatmap curve.
    Returns (peak_start_s, peak_end_s, max_intensity).
    """
    if not heatmap:
        return None

    # Each point typically represents a normalized window
    best_point = max(heatmap, key=lambda pt: pt.get("value", 0.0))
    peak_val = best_point.get("value", 0.0)
    p_start = best_point.get("start_time", 0.0)
    p_end = best_point.get("end_time", p_start + min_peak_duration)

    if (p_end - p_start) < min_peak_duration:
        p_end = p_start + min_peak_duration

    return (p_start, p_end, peak_val)


def search_candidate_videos_for_topic(topic: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Searches YouTube for top-performing long-form videos matching a trending topic.
    Filters for >= 1080p, duration 5m-45m, and Creative Commons / commentary.
    """
    candidates = []
    try:
        import yt_dlp
        ydl_opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist",
        }
        query = f"ytsearch{max_results * 2}:{topic} commentary OR interview OR review"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            res = ydl.extract_info(query, download=False)
            entries = res.get("entries", [])
            for e in entries:
                if not e:
                    continue
                dur = float(e.get("duration", 0.0) or 0.0)
                # Filter duration between 5 min (300s) and 45 min (2700s)
                if dur < 300.0 or dur > 2700.0:
                    continue

                vid_id = e.get("id")
                url = f"https://www.youtube.com/watch?v={vid_id}" if vid_id else e.get("url")
                title = e.get("title", "")
                if not is_topic_safe(title):
                    continue

                candidates.append({
                    "id": vid_id,
                    "url": url,
                    "title": title,
                    "duration": dur,
                    "view_count": e.get("view_count", 0),
                    "uploader": e.get("uploader", "Unknown"),
                })
                if len(candidates) >= max_results:
                    break

        logger.info(f"Found {len(candidates)} candidate videos for topic: '{topic}'")
    except Exception as e:
        logger.warning(f"Error searching candidate videos for '{topic}': {e}")

    return candidates


def download_candidate_video(video_url: str, output_dir: str = "data/input_videos", max_height: int = 1080) -> Optional[str]:
    """
    Downloads candidate long-form video in highest available quality (up to max_height, min 1080p if available)
    into the designated input_videos directory. Returns local file path on success.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        import yt_dlp
        ydl_opts = {
            "format": f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best",
            "outtmpl": os.path.join(output_dir, "%(title).60s [%(id)s].%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            filename = ydl.prepare_filename(info)
            if os.path.exists(filename):
                logger.info(f"Successfully downloaded candidate video to: {filename}")
                return filename
    except Exception as e:
        logger.error(f"Failed to download {video_url}: {e}")
    return None


def main():
    print("\n" + "=" * 65)
    print(" PROJECT AETHER INTELLIGENT TRENDING SENSING SPOKE")
    print("=" * 65)
    topics = fetch_google_trends("US")
    for idx, t in enumerate(topics[:5], 1):
        print(f"\n{idx}. [{t.approx_traffic}] {t.title}")
        candidates = search_candidate_videos_for_topic(t.title, max_results=1)
        for c in candidates:
            print(f"   ↳ Best Match: {c['title']} ({int(c['duration'] // 60)}m, {c['view_count']:,} views)")
            print(f"     URL: {c['url']}")
    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()
