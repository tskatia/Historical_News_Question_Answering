import os
from typing import Optional


class LlamaWrapper:
    """Wrapper around HuggingFace Causal Language Models (e.g. LLaMA / Qwen) for local RAG inference."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        load_in_4bit: bool = True,
        device_map: str = "auto"
    ):
        if model_name is None:
            model_name = os.environ.get("LLM_MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map

        self.tokenizer = None
        self.model = None
        self._is_loaded = False

    def _ensure_loaded(self):
        """Lazy load tokenizer and model to prevent blocking initialization."""
        if self._is_loaded:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        model_kwargs = {
            "trust_remote_code": True,
        }

        # Check CUDA availability and bitsandbytes support
        if torch.cuda.is_available():
            model_kwargs["device_map"] = self.device_map
            if self.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                except ImportError:
                    model_kwargs["torch_dtype"] = torch.float16
            else:
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self._is_loaded = True

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """Generate response text for a given prompt."""
        self._ensure_loaded()
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Slice off prompt tokens to return only the new response
        input_length = inputs["input_ids"].shape[1]
        response_tokens = output_ids[0][input_length:]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)
