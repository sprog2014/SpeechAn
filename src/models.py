import os
import threading
import logging
import warnings
import gigaam
import torch
from transformers import AutoTokenizer, TextIteratorStreamer
from optimum.intel.openvino import OVModelForCausalLM

# Подавляем FutureWarning от torch/gigaam
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Глобальные переменные
_asr_model = None
_asr_lock = threading.Lock()

_vad_model = None
_vad_utils = None
_vad_lock = threading.Lock()

_llm_model = None
_llm_tokenizer = None
_llm_lock = threading.Lock()

def get_asr_model():
    global _asr_model
    with _asr_lock:
        if _asr_model is None:
            logger.info("Initializing GigaAM ASR model (v3_e2e_ctc)...")
            try:
                _asr_model = gigaam.load_model('v3_e2e_ctc')
                _asr_model.eval()
                logger.info("GigaAM ASR model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load GigaAM ASR model: {e}")
                raise
    return _asr_model

def get_vad_model():
    global _vad_model, _vad_utils
    with _vad_lock:
        if _vad_model is None:
            logger.info("Initializing Silero VAD model...")
            try:
                model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=False,
                                              onnx=False,
                                              trust_repo=True)
                _vad_model = model
                _vad_utils = utils
                logger.info("Silero VAD model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Silero VAD model: {e}")
                raise
    return _vad_model, _vad_utils

class OpenVINOLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def create_chat_completion(self, messages, max_tokens=1000, temperature=0.1, stream=False, **kwargs):
        # Эмулируем интерфейс llama-cpp для минимизации правок в вызывающем коде
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            return streamer

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_ids = output_ids[0][len(inputs.input_ids[0]):]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            'choices': [
                {
                    'message': {
                        'content': response_text
                    }
                }
            ]
        }

    def generate(self, messages, max_new_tokens=1000, temperature=0.1):
        # Сохраняем прямой метод для явного использования
        result = self.create_chat_completion(messages, max_tokens=max_new_tokens, temperature=temperature)
        return result['choices'][0]['message']['content']

def get_llm():
    global _llm_model, _llm_tokenizer
    with _llm_lock:
        if _llm_model is None:
            model_path = os.getenv("LLM_MODEL_PATH", "models/qwen2.5-7b-instruct-ov")

            logger.info(f"Initializing Qwen2.5 OpenVINO model from {model_path}...")

            if os.path.exists(model_path) and os.path.isdir(model_path):
                try:
                    logger.info("Loading model from local path...")
                    _llm_model = OVModelForCausalLM.from_pretrained(
                        model_path,
                        device="CPU",
                        ov_config={"PERFORMANCE_HINT": "LATENCY"}
                    )
                    _llm_tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
                    logger.info("Qwen2.5 OpenVINO model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load Qwen2.5 OpenVINO model from {model_path}: {e}")
                    raise
            else:
                error_msg = f"Model directory not found at {model_path}. Please run setup script or convert model manually."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

    return OpenVINOLLM(_llm_model, _llm_tokenizer)

def get_locked_llm():
    return get_llm()
