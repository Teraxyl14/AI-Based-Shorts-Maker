import json
import redis
import hashlib
from loguru import logger

REDIS_HOST = "localhost"
REDIS_PORT = 6379
TRANSCRIPT_TTL = 3600

def push_transcript_to_redis(
    source_path: str,
    words: list,
    full_text: str,
    language: str,
    avg_confidence: float
):
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    
    redis_key = f"aether:transcript:{hash(source_path)}"
    
    payload = {
        "words": words,
        "full_text": full_text,
        "language": language,
        "avg_confidence": avg_confidence,
        "source_path": source_path,
        "schema_version": "1.0"
    }
    
    r.setex(
        name=redis_key,
        time=TRANSCRIPT_TTL,
        value=json.dumps(payload, ensure_ascii=False)
    )
    
    logger.success(
        f"[IPC BRIDGE] Transcript pushed to Redis key: {redis_key} | "
        f"Words: {len(words)} | Language: {language}"
    )