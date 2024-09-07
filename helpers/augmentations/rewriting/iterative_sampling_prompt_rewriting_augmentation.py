import logging

from helpers.utils import LOGGER_NAME

from .iterative_prompt_rewriting_augmentation import IterativePromptRewritingAugmentation
from .prompt_rewriting_augmentation import REWRITING_PROMPT_FIXED

logger = logging.getLogger(LOGGER_NAME)


class IterativeSamplingPromptRewritingAugmentation(IterativePromptRewritingAugmentation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # kwargs
        self.ensemble_count: int = kwargs.pop("ensemble_count")
        self.model_kwargs = self.model_kwargs | {"temperature": kwargs.pop("sampling_temperature")}

    def augmentation(self, prompt: str) -> str | list[str]:
        prompts = []
        for i in range(self.ensemble_count):
            logger.debug(f"Rewriting version {i + 1}")
            rewritten_prompt = super().augmentation(prompt)
            prompts.append(rewritten_prompt)
        return prompts


ITERATIVE_SAMPLING_PROMPT_REWRITING_AUGMENTATION_CONFIGS = {
    "iterative_sampling_prompt_rewriting-llama3": {
        "class": IterativeSamplingPromptRewritingAugmentation,
        "rewriting_model": "llama-3-8b-instruct",
        "rewriting_prompt": REWRITING_PROMPT_FIXED,
        "perplexity_model": "gpt2",
        "minimize_perplexity": False,
        "iterations": 5,
        "ensemble_count": 5,
        "sampling_temperature": 1.0,
    }
}
