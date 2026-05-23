import torch
import logging
from models import get_emotion_model

logger = logging.getLogger(__name__)

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

    # GigaAMEmo ожидает батч [batch, samples] и длины
    lengths = torch.tensor([audio_chunk.shape[1]])

    with torch.no_grad():
        # В официальном пакете gigaam вызов модели возвращает промежуточные слои
        outputs = model(audio_chunk, lengths)

        # Если это GigaAMEmo из пакета gigaam
        if isinstance(outputs, tuple):
            features = outputs[0]
            # Пулинг по времени (среднее)
            pooled = features.mean(dim=-1)
            logits = model.head(pooled)
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        pred_id = torch.argmax(logits, dim=-1).item()
        probs = torch.softmax(logits, dim=-1)
        confidence = probs[0, pred_id].item()

    # Дефолтные метки для GigaAMEmo
    emotion_labels = {0: 'neutral', 1: 'positive', 2: 'negative', 3: 'angry'}
    emotion = emotion_labels.get(pred_id, f"unknown_{pred_id}")

    return emotion, confidence
