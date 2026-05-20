import torch
import torchaudio
from models import load_emotion_model

def predict_emotion(audio_chunk, sample_rate=16000):
    model, feature_extractor = load_emotion_model()
    # Приводим к тензору, если пришёл массив numpy
    if isinstance(audio_chunk, torch.Tensor):
        waveform = audio_chunk
    else:
        waveform = torch.tensor(audio_chunk).unsqueeze(0)
    inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1)[0, pred_id].item()
    emotion_labels = ["neutral", "positive", "negative", "other", "speech"]
    emotion = emotion_labels[pred_id]
    return emotion, confidence