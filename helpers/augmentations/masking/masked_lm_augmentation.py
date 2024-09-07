import logging

from transformers import pipeline

from helpers.augmentations.perplexity_wir_augmentation import PerplexityWIRAugmentation
from helpers.utils import LOGGER_NAME, detokenize

logger = logging.getLogger(LOGGER_NAME)


class MaskedLMAugmentation(PerplexityWIRAugmentation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # kwargs
        self.mask_fill_model: str = kwargs.pop("mask_fill_model")
        self.mask_token: str = kwargs.pop("mask_token")

        # init
        self.mask_fill_pipeline = pipeline("fill-mask", self.mask_fill_model)  # type: ignore

        self.print_augmentation_config()

    def perplexity_wir_augmentation(self, words: list[str], word_idx: int) -> list[str]:
        unmasked = words[word_idx]
        words[word_idx] = self.mask_token
        masked_prompt = detokenize(words)
        logger.debug(f"Masked prompt: {repr(masked_prompt)}")

        raw_candidates = self.mask_fill_pipeline(masked_prompt, top_k=self.top_k)
        candidates = [x["token_str"].strip() for x in raw_candidates]  # type: ignore
        logger.debug(f"Candidates for {repr(unmasked)}: {candidates}")
        return candidates


MASKED_LM_AUGMENTATION_CONFIGS = {
    "masked_lm-beam_search-distilroberta": {
        "class": MaskedLMAugmentation,
        "search_method": "beam_search",
        "beam_width": 5,
        "perplexity_model": "gpt2",
        "iterations": 10,
        "top_k": 5,
        "mask_fill_model": "distilroberta-base",
        "mask_token": "<mask>",
    },
    "masked_lm-greedy_search-distilroberta": {
        "class": MaskedLMAugmentation,
        "search_method": "greedy_search",
        "early_stopping": True,
        "perplexity_model": "gpt2",
        "iterations": 10,
        "top_k": 5,
        "mask_fill_model": "distilroberta-base",
        "mask_token": "<mask>",
    },
}
