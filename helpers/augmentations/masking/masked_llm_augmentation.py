import inspect
import logging

from helpers.augmentations.perplexity_wir_augmentation import PerplexityWIRAugmentation
from helpers.models import create_model
from helpers.utils import LOGGER_NAME, detokenize

logger = logging.getLogger(LOGGER_NAME)

MAX_WORD_LENGTH = 20


class MaskedLLMAugmentation(PerplexityWIRAugmentation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # kwargs
        self.mask_fill_model_name: str = kwargs.pop("mask_fill_model")
        self.mask_fill_prompt: str = kwargs.pop("mask_fill_prompt")
        self.mask_token: str = kwargs.pop("mask_token")

        # init
        self.model_kwargs = {
            "max_new_tokens": 1000,
            "temperature": 0.0,
            "generation_mode": "freeform",
            "system_prompt": None,
        }
        if self.mask_fill_model_name == self.model_name:
            self.mask_fill_model = self.model
        else:
            self.mask_fill_model = create_model(self.mask_fill_model_name, model_context="mask fill", **self.model_kwargs)  # fmt: off

        self.print_augmentation_config()
        logger.info(f"\n{self.mask_fill_prompt}")

    def perplexity_wir_augmentation(self, words: list[str], word_idx: int) -> list[str]:
        unmasked = words[word_idx]
        words[word_idx] = self.mask_token
        masked_prompt = detokenize(words)
        logger.debug(f"Masked prompt: {repr(masked_prompt)}")

        mask_fill_prompt = (
            self.mask_fill_prompt.replace("{top_k}", str(self.top_k))
            .replace("{mask_token}", self.mask_token)
            .replace("{prompt}", masked_prompt)
        )
        raw_candidates = self.mask_fill_model.predict(mask_fill_prompt, **self.model_kwargs)
        candidates = [x.strip() for x in raw_candidates.split(",")]
        candidates = [x for x in candidates if len(x) <= MAX_WORD_LENGTH]
        logger.debug(f"Candidates for {repr(unmasked)}: {candidates}")
        return candidates

    def clear_memory(self):
        self.mask_fill_model.clear_memory()
        return super().clear_memory()


MASK_FILL_PROMPT = inspect.cleandoc('''
Given a sentence containing one word masked with the {mask_token} mask token, respond with the top {top_k} candidates for the masked word. Do not answer with anything other than the candidates for the masked word.

Here are three examples:

Example sentence 0: """Identify whether the given pair of sentences {mask_token} entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':"""
Example answer 0: demonstrate, indicate, represent, show, suggest

Example sentence 1: """{mask_token} if the given pair of sentences displays entailment, neutral, or contradiction. Respond with 'entailment', 'neutral', or 'contradiction':"""
Example answer 1: Determine, Evaluate, Identify, Asses, Decide

Example sentence 2: """Examine the pair of sentences and determine if they exhibit entailment, neutral, or contradiction. {mask_token} with either 'entailment', 'neutral', or 'contradiction':"""
Example answer 2: Answer, Respond, Reply, Return, Output

Here is the real sentence:

Sentence: """{prompt}"""
Answer:
''')

MASKED_LLM_AUGMENTATION_CONFIGS = {
    "masked_llm-beam_search-llama3": {
        "class": MaskedLLMAugmentation,
        "search_method": "beam_search",
        "beam_width": 5,
        "perplexity_model": "gpt2",
        "iterations": 10,
        "top_k": 5,
        "mask_fill_model": "llama-3-8b-instruct",
        "mask_fill_prompt": MASK_FILL_PROMPT,
        "mask_token": "<mask>",
    },
    "masked_llm-greedy_search-llama3": {
        "class": MaskedLLMAugmentation,
        "search_method": "greedy_search",
        "early_stopping": True,
        "perplexity_model": "gpt2",
        "iterations": 10,
        "top_k": 5,
        "mask_fill_model": "llama-3-8b-instruct",
        "mask_fill_prompt": MASK_FILL_PROMPT,
        "mask_token": "<mask>",
    },
}
