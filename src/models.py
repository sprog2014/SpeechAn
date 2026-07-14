import os
import threading
import logging
import warnings
import gigaam
import torch

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

class LlamaWrapper:
    def __init__(self, llama_model):
        self.model = llama_model

    def create_chat_completion(self, messages, max_tokens=2048, temperature=0.1, stream=False, **kwargs):
        if stream:
            # Возвращаем генератор, выдающий только текстовые фрагменты
            response_iter = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            def chunk_generator():
                for chunk in response_iter:
                    choices = chunk.get('choices', [])
                    if choices:
                        delta = choices[0].get('delta', {})
                        if 'content' in delta:
                            yield delta['content']
            return chunk_generator()
        else:
            return self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                **kwargs
            )

    def generate(self, messages, max_new_tokens=2048, temperature=0.1):
        result = self.create_chat_completion(messages, max_tokens=max_new_tokens, temperature=temperature)
        return result['choices'][0]['message']['content']

def get_llm():
    global _llm_model
    with _llm_lock:
        if _llm_model is None:
            import llama_cpp
            from db_utils import get_system_setting

            active_model = get_system_setting('active_model', 'q4_k_m')

            if active_model == 'q8_0':
                model_filename = "qwen2.5-7b-instruct-q8_0.gguf"
            else:
                model_filename = "qwen2.5-7b-instruct-q4_k_m.gguf"

            possible_paths = [
                f"models/{model_filename}",
                f"/opt/calls/models/{model_filename}",
                f"../models/{model_filename}"
            ]

            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if not model_path:
                # Fallback к любому файлу .gguf в models/
                if os.path.exists("models"):
                    for f in os.listdir("models"):
                        if f.endswith(".gguf"):
                            model_path = os.path.join("models", f)
                            break

            if not model_path:
                model_path = f"models/{model_filename}"

            logger.info(f"Loading Llama model from {model_path} with n_ctx=16384, n_threads=20...")

            try:
                llama_instance = llama_cpp.Llama(
                    model_path=model_path,
                    n_ctx=16384,
                    n_threads=20,
                    verbose=False
                )
                _llm_model = LlamaWrapper(llama_instance)
                logger.info("Llama model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Llama model from {model_path}: {e}")
                raise

    return _llm_model

def get_locked_llm():
    return get_llm()
