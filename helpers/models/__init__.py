from .bert_model import BERTModel
from .llama_model import LlamaModel
from .model import EXCLUDED_LOG_KEYS, Model
from .t5_model import T5Model


def create_model(model_name: str, **kwargs) -> Model:
    if model_name == "flan-t5-large":
        return T5Model("google/flan-t5-large", **kwargs)

    if model_name == "llama-3-8b-instruct":
        return LlamaModel("meta-llama/Meta-Llama-3-8B-Instruct", **kwargs)

    if model_name == "llama-3-8b":
        return LlamaModel("meta-llama/Meta-Llama-3-8B", **kwargs)

    if model_name == "llama-2-13b-chat":
        return LlamaModel("meta-llama/Llama-2-13b-chat-hf", **kwargs)

    if model_name == "distilroberta-base":
        return BERTModel("distilbert/distilroberta-base", **kwargs)

    if "llama-3-8b-instruct-sft" in model_name:
        return LlamaModel(model_name, **kwargs)

    if "flan-t5-large-sft" in model_name:
        return T5Model(model_name, **kwargs)

    raise ValueError(f"Model {model_name} not found")


__all__ = ["create_model", "EXCLUDED_LOG_KEYS"]
