import os
import sys
import time
import librosa
import numpy as np
import openvino_genai as ov_genai
from pathlib import Path
from loguru import logger
from ipc_bridge import push_transcript_to_redis

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

# Use the absolute path to the M: drive where we exported the model
MODEL_PATH = Path("M:/ProjectAether/wsl2_gpu/models/whisper-base-int4-ov")
DEVICE = "CPU"           
SAMPLE_RATE = 16000      
CHUNK_DURATION = 30      
CHUNK_OVERLAP = 2        

def read_wav_normalized(audio_path: str) -> np.ndarray:
    raw, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    logger.info(f"[AUDIO] Loaded: {audio_path} | Duration: {len(raw)/SAMPLE_RATE:.1f}s")
    return raw.astype(np.float32)

def extract_audio_from_video(video_path: str) -> str:
    import subprocess
    audio_path = video_path.replace(".mp4", "_audio.wav")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                      
        "-acodec", "pcm_s16le",     
        "-ar", str(SAMPLE_RATE),    
        "-ac", "1",                 
        audio_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    logger.success(f"[AUDIO] Extracted to: {audio_path}")
    return audio_path

def load_npu_whisper_pipeline() -> ov_genai.WhisperPipeline:
    logger.info(f"[NPU] Loading Whisper INT4 model from: {MODEL_PATH}")
    
    # 1. Configure the NPU Pipeline properties
    ov_config = {}
    ov_config["PERFORMANCE_HINT"] = "LATENCY"
    
    # CRITICAL FIX: Force the NPU to allocate the expanded {1, 447} input 
    # tensor required for word-level attention masking during initialization.
    ov_config["word_timestamps"] = True
    
    # 2. Initialize the pipeline with the config unpacked
    pipe = ov_genai.WhisperPipeline(
        str(MODEL_PATH),
        DEVICE,
        **ov_config
    )
    
    logger.success(f"[NPU] Whisper pipeline loaded on {DEVICE}")
    return pipe

def transcribe_chunked_with_word_timestamps(
    pipe: ov_genai.WhisperPipeline,
    raw_audio: np.ndarray,
    language: str = "en"
) -> dict:
    total_samples = len(raw_audio)
    chunk_samples = CHUNK_DURATION * SAMPLE_RATE
    overlap_samples = CHUNK_OVERLAP * SAMPLE_RATE
    
    all_words = []
    full_text_parts = []
    chunk_idx = 0
    offset_sec = 0.0
    start_sample = 0
    
    while start_sample < total_samples:
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk = raw_audio[start_sample:end_sample]  
        
        logger.info(f"[NPU] Transcribing chunk {chunk_idx+1} | Offset: {offset_sec:.1f}s")
        t0 = time.perf_counter()
        
        # ── Core OpenVINO GenAI Whisper Inference ─────────
        result = pipe.generate(
            chunk,
            task="transcribe",
            return_timestamps=True,     # <--- FIXED: Now a strict Boolean
            max_new_tokens=448,         
        )
        
        elapsed = time.perf_counter() - t0
        rtf = elapsed / (len(chunk) / SAMPLE_RATE)  
        logger.info(f"[NPU] Chunk {chunk_idx+1} complete | {elapsed:.2f}s | RTF: {rtf:.2f}x")
        
        # ── Parse word-level timestamp chunks ─────────────
        if hasattr(result, 'chunks') and result.chunks:
            for chunk_result in result.chunks:
                word = chunk_result.text.strip()
                if not word:
                    continue
                word_entry = {
                    "word": word,
                    "start": round(chunk_result.start_ts + offset_sec, 3),
                    "end": round(chunk_result.end_ts + offset_sec, 3),
                    "confidence": getattr(chunk_result, 'probability', 0.95)
                }
                all_words.append(word_entry)
        
        full_text_parts.append(str(result).strip())
        start_sample += chunk_samples - overlap_samples
        offset_sec += CHUNK_DURATION - CHUNK_OVERLAP
        chunk_idx += 1
    
    all_words = deduplicate_overlap_words(all_words)
    avg_confidence = (sum(w["confidence"] for w in all_words) / len(all_words)) if all_words else 0.0
    
    return {
        "words": all_words,
        "full_text": " ".join(full_text_parts),
        "language": language,
        "avg_confidence": round(avg_confidence, 4)
    }

def deduplicate_overlap_words(words: list) -> list:
    if len(words) < 2: return words
    deduped = [words[0]]
    for word in words[1:]:
        prev = deduped[-1]
        if (word["word"].lower() == prev["word"].lower() and abs(word["start"] - prev["start"]) < 0.1):
            continue
        deduped.append(word)
    return deduped

def run_transcription_agent(video_path: str, language: str = "en"):
    logger.info(f"[AGENT] Starting NPU Transcription Agent | Source: {video_path}")
    audio_path = extract_audio_from_video(video_path)
    raw_audio = read_wav_normalized(audio_path)
    pipe = load_npu_whisper_pipeline()
    
    t_start = time.perf_counter()
    transcript = transcribe_chunked_with_word_timestamps(pipe, raw_audio, language)
    t_total = time.perf_counter() - t_start
    
    logger.success(f"[AGENT] Transcription complete | Words: {len(transcript['words'])} | Time: {t_total:.1f}s")
    
    push_transcript_to_redis(
        source_path=video_path,
        words=transcript["words"],
        full_text=transcript["full_text"],
        language=transcript["language"],
        avg_confidence=transcript["avg_confidence"]
    )
    
    os.unlink(audio_path)
    logger.info("[AGENT] NPU Transcription Agent complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python transcription_agent.py <video_path>")
        sys.exit(1)
    video_file = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    run_transcription_agent(video_file, lang)