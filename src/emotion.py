import torch
import logging
from collections import Counter
from models import get_emotion_model

logger = logging.getLogger(__name__)

def predict_emotion(waveform: torch.Tensor, sample_rate: int = 16000, chunk_sec: int = 10, overlap_sec: int = 2):
    """
    Принимает mono аудио тензор, нарезает на куски, анализирует каждый
    и возвращает наиболее часто встречающуюся эмоцию.
    """
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    total_samples = waveform.shape[0]
    chunk_samples = chunk_sec * sample_rate
    overlap_samples = overlap_sec * sample_rate
    step_samples = chunk_samples - overlap_samples

    # Минимум 400 отсчетов (25мс при 16кГц) для работы STFT
    MIN_SAMPLES = 400

    if total_samples < MIN_SAMPLES:
        return "neutral"

    model = get_emotion_model()
    emotions_found = []

    start_samp = 0
    while start_samp < total_samples:
        end_samp = min(start_samp + chunk_samples, total_samples)
        chunk = waveform[start_samp:end_samp]

        if len(chunk) >= MIN_SAMPLES:
            # Подготовка для модели
            input_tensor = chunk.unsqueeze(0)  # [1, samples]
            lengths = torch.tensor([input_tensor.shape[1]])

            with torch.no_grad():
                outputs = model(input_tensor, lengths)

                # Логика извлечения логитов (из оригинального emotion.py)
                if isinstance(outputs, tuple):
                    features = outputs[0]
                    pooled = features.mean(dim=-1)
                    logits = model.head(pooled)
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                pred_id = torch.argmax(logits, dim=-1).item()

                emotion_labels = {0: 'neutral', 1: 'positive', 2: 'negative', 3: 'angry'}
                emotions_found.append(emotion_labels.get(pred_id, 'neutral'))

        if end_samp == total_samples:
            break
        start_samp += step_samples

    if not emotions_found:
        return "neutral"

    # Возвращаем самую частую эмоцию
    most_common = Counter(emotions_found).most_common(1)[0][0]
    logger.info(f"Emotion analysis finished. Samples: {len(emotions_found)}, Winner: {most_common}")
    return most_common
