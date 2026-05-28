import torch
import logging
import numpy as np
from models import get_emotion_model

logger = logging.getLogger(__name__)

def predict_emotion(audio_chunk, sample_rate=16000):
    """
    Принимает numpy-массив или torch.Tensor (1D), возвращает (emotion_label, confidence).
    """
    MIN_SAMPLES = 400
    if len(audio_chunk) < MIN_SAMPLES:
        return "neutral", 1.0

    try:
        model = get_emotion_model()
        if not isinstance(audio_chunk, torch.Tensor):
            audio_chunk = torch.tensor(audio_chunk, dtype=torch.float32)

        if audio_chunk.ndim == 1:
            audio_chunk = audio_chunk.unsqueeze(0)

        lengths = torch.tensor([audio_chunk.shape[1]])

        with torch.no_grad():
            outputs = model(audio_chunk, lengths)
            if isinstance(outputs, tuple):
                features = outputs[0]
                pooled = features.mean(dim=-1)
                logits = model.head(pooled)
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(logits, dim=-1).item()
            confidence = probs[0, pred_id].item()

        emotion_labels = {0: 'neutral', 1: 'positive', 2: 'negative', 3: 'angry'}
        emotion = emotion_labels.get(pred_id, f"unknown_{pred_id}")

        return emotion, confidence, probs[0].tolist()

    except Exception as e:
        logger.error(f"Error during emotion prediction: {e}")
        return "neutral", 0.0, [0.0]*4

def predict_emotions_full(audio_data, sample_rate=16000, chunk_sec=5):
    """
    Анализирует всё аудио целиком, нарезая на куски и усредняя вероятности.
    Возвращает словарь с финальной эмоцией и вероятностями.
    """
    logger.info(f"Analyzing full audio emotions ({len(audio_data)} samples)")

    chunk_samples = int(chunk_sec * sample_rate)
    total_samples = len(audio_data)

    all_probs = []

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio_data[start:end]
        if len(chunk) < 400:
            continue

        _, _, probs = predict_emotion(chunk, sample_rate)
        all_probs.append(probs)

    if not all_probs:
        return {"emotion": "neutral", "confidence": 1.0, "probs": {}}

    # Усредняем вероятности по всем кускам
    avg_probs = np.mean(all_probs, axis=0)
    pred_id = np.argmax(avg_probs)

    emotion_labels = {0: 'neutral', 1: 'positive', 2: 'negative', 3: 'angry'}
    final_emotion = emotion_labels.get(pred_id, "unknown")

    return {
        "emotion": final_emotion,
        "confidence": float(avg_probs[pred_id]),
        "all_probs": {emotion_labels[i]: float(avg_probs[i]) for i in range(len(avg_probs)) if i in emotion_labels}
    }
