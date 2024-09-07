import gc
import logging
import pprint
from abc import ABC

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # type: ignore
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from helpers.utils import LOGGER_NAME, log_vram_usage

logger = logging.getLogger(LOGGER_NAME)

EXCLUDED_LOG_KEYS = set(["model", "tokenizer", "fill_mask_pipeline", "stopwords", "rewriting_model", "compute_perplexity", "emb_tokenizer", "emb_model", "filtered_vocab", "filtered_embeddings", "search_method", "model", "augmented_prompts", "format_input", "format_output", "mask_fill_prompt", "rewriting_prompt", "vocabulary", "vocabulary_embs"])  # fmt: skip


class Model(ABC):
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    model: PreTrainedModel | PeftModel
    task_type: TaskType

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model_context: str = kwargs.pop("model_context")

        self.max_new_tokens: int = kwargs.pop("max_new_tokens")
        self.temperature: float = kwargs.pop("temperature")
        self.generation_mode: str = kwargs.pop("generation_mode")
        self.system_prompt: str = kwargs.pop("system_prompt")

        if self.generation_mode == "constrained":
            self.output_words = kwargs.pop("output_words")

        self.device_map = "auto"
        self.torch_dtype = torch.float16

    def set_logits_processor(self):
        if self.generation_mode == "constrained":
            self.repetition_penalty = 10000.0
            self.output_word_ids = [self.tokenizer.encode(word, add_special_tokens=False) for word in self.output_words]
            self.logits_processor_list = LogitsProcessorList([ConstrainedLogitsProcessor(self.output_word_ids)])
        elif self.generation_mode == "freeform":
            return
        else:
            raise ValueError(f"Generation mode {self.generation_mode} not found")

    def print_model_config(self):
        print_dict = {x: self.__dict__[x] for x in self.__dict__ if x not in EXCLUDED_LOG_KEYS}
        logger.info(f"\n{self.model_context.capitalize()} model config:\n{pprint.pformat(print_dict)}\n")

    def predict(self, input_text: str, **kwargs) -> str:
        raise NotImplementedError

    def clear_memory(self):
        if not hasattr(self, "model"):
            logger.warning(f"{self.model_context.capitalize()} model ({self.model_name}) not found in VRAM")
            return

        delattr(self, "model")
        logger.info(f"Deleted {self.model_context} model ({self.model_name}) from VRAM")

        gc.collect()
        torch.cuda.empty_cache()

        log_vram_usage()

    def convert_to_peft(self, peft_config: LoraConfig):
        self.model = get_peft_model(self.model, peft_config)  # type: ignore
        trainable_params, all_param = self.model.get_nb_trainable_parameters()
        logger.info(f"Trainable parameters: {trainable_params:,d} || All parameters: {all_param:,d} || %: {100 * trainable_params / all_param:.4f}")  # fmt: skip
        gc.collect()

    def load_adapter(self, adapter_name: str):
        self.model.load_adapter(adapter_name)  # type: ignore

    def __hash__(self):
        return hash(self.model_name)


class ConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, output_word_ids: list[list[int]]):
        self.output_word_ids = output_word_ids
        self.generated_tokens = []
        self.idx = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.ones(scores.shape).to(scores.device)
        allowed_tokens = [
            output_word_id[self.idx]
            for output_word_id in self.output_word_ids
            if self.idx < len(output_word_id) and self.generated_tokens == output_word_id[: self.idx]
        ]
        mask[:, allowed_tokens] = 0
        scores = scores.masked_fill(mask.bool(), -float("inf"))  # type: ignore
        self.generated_tokens.append(torch.argmax(scores, dim=-1).item())
        self.idx += 1
        return scores

    def reset(self) -> None:
        self.idx = 0
        self.generated_tokens = []

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return self.__repr__()
