from models import get_whisper

def transcribe_audio(audio_path: str, language: str = "ru"):
    """Возвращает список сегментов: [(start, end, text), ...]"""
    model = get_whisper()
    segments, _ = model.transcribe(audio_path, language=language, beam_size=5)
    return [(seg.start, seg.end, seg.text.strip()) for seg in segments]
