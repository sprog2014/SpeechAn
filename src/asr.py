import logging
import time
from models import get_whisper

logger = logging.getLogger(__name__)

def transcribe_audio(audio_path: str, language: str = "ru"):
    """Возвращает список сегментов: [(start, end, text), ...]"""
    start_time = time.time()
    logger.info(f"Starting transcription for {audio_path}")

    model = get_whisper()
    # vad_filter=True обеспечивает абсолютные таймстампы от начала файла
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    result = []
    for seg in segments:
        result.append((seg.start, seg.end, seg.text.strip()))

    duration = time.time() - start_time
    logger.info(f"Transcription finished: {len(result)} segments found in {duration:.2f}s (lang: {info.language})")
    return result
