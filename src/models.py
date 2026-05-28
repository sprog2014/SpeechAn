import os
import threading
import logging
import warnings
import gigaam
import torch
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

# Подавляем FutureWarning от torch/gigaam
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Глобальные переменные
_asr_model = None
_asr_lock = threading.Lock()

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

class OpenVINOLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def create_chat_completion(self, messages, max_tokens=1000, temperature=0.1, **kwargs):
        # Эмулируем интерфейс llama-cpp для минимизации правок в вызывающем коде
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")

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
            model_id = "Qwen/Qwen2.5-7B-Instruct"
            model_path = os.getenv("LLM_MODEL_PATH", "models/qwen2.5-7b-instruct-ov")

            logger.info(f"Initializing Qwen2.5 OpenVINO model from {model_path}...")

            try:
                # Пытаемся загрузить локально
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    logger.info("Loading model from local path...")
                    _llm_model = OVModelForCausalLM.from_pretrained(
                        model_path,
                        device="CPU",
                        ov_config={"PERFORMANCE_HINT": "LATENCY"}
                    )
                    _llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
                else:
                    logger.info(f"Model not found at {model_path}. Exporting from Hugging Face {model_id}...")
                    # Экспортируем и сохраняем
                    _llm_model = OVModelForCausalLM.from_pretrained(
                        model_id,
                        export=True,
                        quantization_config={"bits": 8},
                        device="CPU"
                    )
                    _llm_tokenizer = AutoTokenizer.from_pretrained(model_id)

                    logger.info(f"Saving exported model to {model_path}...")
                    _llm_model.save_pretrained(model_path)
                    _llm_tokenizer.save_pretrained(model_path)

                logger.info("Qwen2.5 OpenVINO model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Qwen2.5 OpenVINO model: {e}")
                raise

    return OpenVINOLLM(_llm_model, _llm_tokenizer)

def get_locked_llm():
    return get_llm()
