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
        # Предполагаемый вызов: модель возвращает логиты
        outputs = model(audio_chunk)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1)[0, pred_id].item()

    emotion_labels = model.config.id2label  # словарь {0: 'neutral', ...}
    emotion = emotion_labels[pred_id]
    return emotion, confidence
