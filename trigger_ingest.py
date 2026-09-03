"""
Project Aether — Manual Ingest Trigger

Publishes a video.ingest event to NATS JetStream to manually kick off
the pipeline for a specific video file.

Usage:
    python trigger_ingest.py <path_to_video>
    python trigger_ingest.py                     # defaults to /data/input_videos/input.mp4
"""
import asyncio
import json
import os
import sys
import time

import nats


async def main():
    # Resolve video path from CLI arg or default
    if len(sys.argv) > 1:
        source_path = sys.argv[1]
    else:
        source_path = "/data/input_videos/input.mp4"

    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    nc = await nats.connect(nats_url)
    js = nc.jetstream()

    payload = {
        "source_path": source_path,
        "file_size": 0,
        "timestamp": time.time(),
    }

    await js.publish("video.ingest", json.dumps(payload).encode("utf-8"))
    print(f"Published ingest event for: {source_path}")
    await nc.close()


if __name__ == "__main__":
    asyncio.run(main())
