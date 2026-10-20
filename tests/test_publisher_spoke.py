"""
Unit and integration test suite for wsl2_gpu.publisher_spoke enterprise upgrades:
- Metadata normalization and length enforcement
- ASCII control character stripping and tag deduplication
- Mandatory AI Synthetic Media disclosure (containsSyntheticMedia: True)
- Automated playlist routing and caption attachment
- State reconciliation guard tracking
"""

import os
import json
import time
import pytest
from unittest.mock import MagicMock

from wsl2_gpu.publisher_spoke import (
    remove_control_characters,
    normalize_tags,
    normalize_and_validate_metadata,
    generate_viral_seo_metadata,
    generate_syndication_package,
    publish_short,
    add_video_to_playlist,
    attach_captions,
    MAX_TITLE_LENGTH,
    MAX_DESCRIPTION_LENGTH,
    MAX_TAGS_LENGTH,
    MAX_TAGS_COUNT,
)


class TestMetadataSanitization:
    """Verifies compliance with YouTube Data API v3 and policy requirements."""

    def test_remove_control_characters(self):
        dirty = "Secret\x00Document\x1fSwap\x7f\tWith\nNewlines"
        clean = remove_control_characters(dirty)
        assert "\x00" not in clean
        assert "\x1f" not in clean
        assert "\x7f" not in clean
        assert "\t" in clean
        assert "\n" in clean
        assert clean.startswith("SecretDocumentSwap")

    def test_normalize_tags_dedup_and_caps(self):
        raw = ["#viral", "VIRAL", '"movieclips"', "scene", "shorts", "#cinema", "scene"]
        tags = normalize_tags(raw)
        # Should deduplicate case-insensitively and strip '#' and quotes
        assert "viral" in tags
        assert "VIRAL" not in tags
        assert "movieclips" in tags
        assert '"movieclips"' not in tags
        assert tags.count("scene") == 1
        assert len(tags) <= MAX_TAGS_COUNT

    def test_normalize_tags_total_byte_limit(self):
        long_tags = [f"tag_number_{i:04d}_with_excessive_padding" for i in range(50)]
        tags = normalize_tags(long_tags)
        total_len = sum(len(t) for t in tags) + len(tags) - 1
        assert total_len <= MAX_TAGS_LENGTH
        assert len(tags) <= MAX_TAGS_COUNT

    def test_normalize_and_validate_metadata_truncation(self):
        huge_title = "A" * 150
        huge_desc = "B" * 6000
        seo_input = {
            "title": huge_title,
            "description": huge_desc,
            "tags": ["cinema", "acting", "drama"],
            "category_id": "invalid_cat"
        }
        sanitized, warnings = normalize_and_validate_metadata(seo_input)
        assert len(sanitized["title"]) <= MAX_TITLE_LENGTH
        assert sanitized["title"].endswith("...")
        assert len(sanitized["description"]) <= MAX_DESCRIPTION_LENGTH
        assert sanitized["category_id"] == "24"  # Default fallback
        assert len(warnings) > 0


class TestSyndicationAndPublishing:
    """Verifies syndication packages and publishing workflows."""

    def test_generate_viral_seo_metadata(self):
        meta = generate_viral_seo_metadata("Secret Document Swap", summary="Jimmy switches numbers")
        assert len(meta["title"]) <= MAX_TITLE_LENGTH
        assert "#shorts" in meta["title"]
        assert meta["category_id"] == "24"
        assert len(meta["tags"]) >= 3

    def test_generate_syndication_package_with_playlist(self):
        pkg = generate_syndication_package(
            "Catch-22 Tomato Scene",
            summary="Milo bargains for olive oil",
            playlist_id="PL_TEST_PLAYLIST_123"
        )
        assert "youtube" in pkg
        assert "tiktok" in pkg
        assert "instagram_reels" in pkg
        assert pkg["youtube"]["playlist_id"] == "PL_TEST_PLAYLIST_123"
        assert pkg["youtube"]["contains_synthetic_media"] is True
        assert len(pkg["tiktok"]["caption"]) <= 150

    def test_publish_short_dry_run_includes_enterprise_fields(self, tmp_path, monkeypatch):
        # Isolate history file to tmp_path so test is 100% hermetic
        isolated_history = tmp_path / "test_pub_history.json"
        monkeypatch.setattr("wsl2_gpu.publisher_spoke.HISTORY_FILE", str(isolated_history))
        monkeypatch.setattr("wsl2_gpu.publisher_spoke.get_todays_upload_count", lambda: 0)

        dummy_video = tmp_path / "test_short.mp4"
        dummy_video.write_bytes(b"\x00" * 1024)

        dummy_srt = tmp_path / "test_short.srt"
        dummy_srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nHello world\n")

        res = publish_short(
            str(dummy_video),
            raw_title="Courtroom Climax",
            summary="Chuck loses his cool",
            srt_path=str(dummy_srt),
            playlist_id="PL_ENTERPRISE_99",
            privacy_status="private",
            notify_subscribers=False
        )
        assert res["status"] == "success"
        assert res["mode"] == "dry_run"
        assert res["playlist_id"] == "PL_ENTERPRISE_99"
        assert res["contains_synthetic_media"] is True
        assert res["notify_subscribers"] is False
        assert res["srt_path"] == str(dummy_srt)

    def test_publish_short_quota_held(self, tmp_path, monkeypatch):
        dummy_video = tmp_path / "test_short.mp4"
        dummy_video.write_bytes(b"\x00" * 1024)
        monkeypatch.setattr("wsl2_gpu.publisher_spoke.get_todays_upload_count", lambda: 4)

        res = publish_short(str(dummy_video), raw_title="Over Quota Short")
        assert res["status"] == "quota_held"
        assert res["daily_uploads"] == 4

    def test_add_video_to_playlist_mock(self):
        mock_youtube = MagicMock()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = {"id": "playlist_item_123"}
        mock_youtube.playlistItems().insert.return_value = mock_insert

        success = add_video_to_playlist(mock_youtube, "vid_abc_123", "PL_TARGET_456")
        assert success is True
        mock_youtube.playlistItems().insert.assert_called_once()

    def test_attach_captions_missing_file(self):
        mock_youtube = MagicMock()
        # Non-existent file should gracefully return False without crashing
        success = attach_captions(mock_youtube, "vid_123", "non_existent_file.srt")
        assert success is False
