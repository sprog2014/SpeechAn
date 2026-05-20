from models import load_whisper

def transcribe_audio(audio_path, language='ru'):
    """Возвращает список сегментов: (start, end, text)"""
    model = load_whisper()
    segments, _ = model.transcribe(audio_path, language=language, beam_size=5)
    result = []
    for seg in segments:
        result.append((seg.start, seg.end, seg.text.strip()))
    return result