import logging

from helpers.utils import LOGGER_NAME

from .prompt_rewriting_augmentation import REWRITING_PROMPT_FIXED, PromptRewritingAugmentation

logger = logging.getLogger(LOGGER_NAME)


class IterativePromptRewritingAugmentation(PromptRewritingAugmentation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # kwargs
        self.iterations: int = kwargs.pop("iterations")

    def augmentation(self, prompt: str) -> str | list[str]:
        for i in range(self.iterations):
            logger.debug(f"Rewriting iteration {i + 1}")
            rewritten_prompt = super().augmentation(prompt)
            if rewritten_prompt == prompt:
                logger.debug("No changes made to the prompt, stopping")
                break
            prompt = rewritten_prompt  # type: ignore
        return prompt


ITERATIVE_PROMPT_REWRITING_AUGMENTATION_CONFIGS = {
    "iterative_prompt_rewriting-llama3": {
        "class": IterativePromptRewritingAugmentation,
        "rewriting_model": "llama-3-8b-instruct",
        "rewriting_prompt": REWRITING_PROMPT_FIXED,
        "perplexity_model": "gpt2",
        "minimize_perplexity": False,
        "iterations": 5,
    },
    "iterative_prompt_rewriting-llama3-sft-zs-promptbench-attackaware": {
        "class": IterativePromptRewritingAugmentation,
        "rewriting_model": "aryanagrawal1/llama-3-8b-instruct-sft-rewriting-zs-promptbench-attackaware",
        "rewriting_prompt": REWRITING_PROMPT_FIXED,
        "perplexity_model": "gpt2",
        "minimize_perplexity": False,
        "iterations": 5,
    },
    "iterative_prompt_rewriting-llama3-sft-zs-promptbench-attackblind": {
        "class": IterativePromptRewritingAugmentation,
        "rewriting_model": "aryanagrawal1/llama-3-8b-instruct-sft-rewriting-zs-promptbench-attackblind",
        "rewriting_prompt": REWRITING_PROMPT_FIXED,
        "perplexity_model": "gpt2",
        "minimize_perplexity": False,
        "iterations": 5,
    },
    "iterative_prompt_rewriting-llama3-sft-fs-promptbench": {
        "class": IterativePromptRewritingAugmentation,
        "rewriting_model": "aryanagrawal1/llama-3-8b-instruct-sft-rewriting-fs-promptbench",
        "rewriting_prompt": REWRITING_PROMPT_FIXED,
        "perplexity_model": "gpt2",
        "minimize_perplexity": False,
        "iterations": 5,
    },
    "iterative_prompt_rewriting-llama3-sft-fs-promptbench-attackaware": {
        "class": IterativePromptRewritingAugmentation,
        "rewriting_model": "aryanagrawal1/llama-3-8b-instruct-sft-rewriting-fs-promptbench-attackaware",
        "rewriting_prompt": REWRITING_PROMPT_FIXED,
        "perplexity_model": "gpt2",
        "minimize_perplexity": False,
        "iterations": 5,
    },
    "iterative_prompt_rewriting-llama3-sft-fs-promptbench-attackblind": {
        "class": IterativePromptRewritingAugmentation,
        "rewriting_model": "aryanagrawal1/llama-3-8b-instruct-sft-rewriting-fs-promptbench-attackblind",
        "rewriting_prompt": REWRITING_PROMPT_FIXED,
        "perplexity_model": "gpt2",
        "minimize_perplexity": False,
        "iterations": 5,
    },
    "iterative_prompt_rewriting-llama3-sft-fs-advglue": {
        "class": IterativePromptRewritingAugmentation,
        "rewriting_model": "aryanagrawal1/llama-3-8b-instruct-sft-rewriting-fs-advglue",
        "rewriting_prompt": REWRITING_PROMPT_FIXED,
        "perplexity_model": "gpt2",
        "minimize_perplexity": False,
        "iterations": 5,
    },
    "iterative_prompt_rewriting-llama3-sft-fs-advglue-promptbench": {
        "class": IterativePromptRewritingAugmentation,
        "rewriting_model": "aryanagrawal1/llama-3-8b-instruct-sft-rewriting-fs-advglue-promptbench",
        "rewriting_prompt": REWRITING_PROMPT_FIXED,
        "perplexity_model": "gpt2",
        "minimize_perplexity": False,
        "iterations": 5,
    },
}
