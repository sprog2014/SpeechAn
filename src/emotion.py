import torch
from models import get_emotion_model

def predict_emotion(audio_chunk, sample_rate=16000):
    """
    Принимает numpy-массив или torch.Tensor (1D), возвращает (emotion_label, confidence).
    emotion_label: одна из ['neutral','positive','negative','other','speech'].
    """
    model = get_emotion_model()
    if not isinstance(audio_chunk, torch.Tensor):
        audio_chunk = torch.tensor(audio_chunk, dtype=torch.float32)

    if audio_chunk.ndim == 1:
        audio_chunk = audio_chunk.unsqueeze(0)  # [1, samples]

    # GigaAMEmo ожидает батч [batch, samples]
    with torch.no_grad():
        outputs = model(audio_chunk)
        # В зависимости от версии gigaam, вывод может быть разным
        # Обычно это объект с атрибутом logits или просто тензор
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        pred_id = torch.argmax(logits, dim=-1).item()
        probs = torch.softmax(logits, dim=-1)
        confidence = probs[0, pred_id].item()

    # Попытка достать метки из конфига или использовать дефолтные
    if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
        emotion_labels = model.config.id2label
    else:
        # Дефолтные метки для GigaAMEmo если конфиг недоступен
        emotion_labels = {0: 'neutral', 1: 'positive', 2: 'negative', 3: 'other', 4: 'speech'}

    emotion = emotion_labels.get(pred_id, f"unknown_{pred_id}")
    return emotion, confidence
