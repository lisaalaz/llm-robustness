import inspect
import logging

from helpers.augmentations import Augmentation
from helpers.models import create_model
from helpers.utils import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

ESC = '"""'


class EnsemblePromptRewritingAugmentation(Augmentation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # kwargs
        self.rewriting_model_name: str = kwargs.pop("rewriting_model")
        self.rewriting_prompt: str = kwargs.pop("rewriting_prompt")
        self.ensemble_count: int = kwargs.pop("ensemble_count")

        # init
        self.model_kwargs = {
            "max_new_tokens": 1000,
            "temperature": 0.0,
            "generation_mode": "freeform",
            "system_prompt": None,
        }
        if self.rewriting_model_name == self.model_name:
            self.rewriting_model = self.model
        else:
            self.rewriting_model = create_model(self.rewriting_model_name, model_context="rewriting", **self.model_kwargs)  # fmt: off

        self.print_augmentation_config()
        logger.info(f"\n{self.rewriting_prompt}")

    def augmentation(self, prompt: str) -> str | list[str]:
        rewriting_prompt = (
            self.rewriting_prompt.replace("{excluded_words}", str(self.class_labels + ["[UNK]"]))
            .replace("{ensemble_count}", str(self.ensemble_count))
            .replace("{prompt}", prompt)
        )
        output = self.rewriting_model.predict(rewriting_prompt, **self.model_kwargs)

        rewritten_prompts = []
        for line in output.split("\n"):
            line = self.remove_prefix_suffix(line)
            if line:
                rewritten_prompts.append(line)

        if not rewritten_prompts:
            logger.warning("No rewritten prompts, using the original prompt")
            return prompt

        return rewritten_prompts

    def remove_prefix_suffix(self, string: str):
        return string.lstrip("0123456789.—()~-*\"' ").rstrip("\"' ")

    def clear_memory(self):
        self.rewriting_model.clear_memory()
        return super().clear_memory()


ENSEMBLE_REWRITING_PROMPT_SAME_LINE = inspect.cleandoc('''
Given a sentence which could possibly be perturbed through an adversarial attack, respond with the top five candidates for the unperturbed sentence. Do not modify the following words: """{excluded_words}""". Do not answer with anything other than the unperturbed sentences.

Sentence: """Uncovering whether the made coupling of condemns revealed entailment, neutral, or contradiction. Cope with 'entailment', 'neutral', or 'contradiction':"""
Answer: """Identify whether the given pair of sentences demonstrates entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':""", """Determine if the given pair of sentences reveals entailment, neutral, or contradiction. Respond with 'entailment', 'neutral', or 'contradiction':""", """Decide whether the provided sentence pair displays entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':""", """Assess if the presented sentence combination shows entailment, neutral, or contradiction. Reply with 'entailment', 'neutral', or 'contradiction':""", """Evaluate whether the given sentence pairing indicates entailment, neutral, or contradiction. Respond using 'entailment', 'neutral', or 'contradiction':"""

Sentence: """Examine the coupling of condemnation and ascertain if their meanings are identical by responses with 'equivalent' or 'not_equivalent'."""
Answer: """Examine the pair of sentences and decide if their meanings are identical by answering with 'equivalent' or 'not_equivalent'.""", """Analyze the provided sentence pair and decide if their meanings are the same by answering with 'equivalent' or 'not_equivalent'.""", """Assess the relationship between the two sentences and identify if their meanings match by replying with 'equivalent' or 'not_equivalent'.""", """Evaluate the connection between the given sentences and determine if they convey the same meaning by responding with 'equivalent' or 'not_equivalent'.""", """Compare the meanings of the presented sentence pair and indicate if they are identical by answering with 'equivalent' or 'not_equivalent'."""

Sentence: """Is the following two trouble equivalent or not? Meet me with 'equivalent' or 'not_equivalent'."""
Answer: """Are the following two sentences equivalent or not? Answer with 'equivalent' or 'not_equivalent'.""", """Are the following two statements equivalent or not? Respond with 'equivalent' or 'not_equivalent'.""", """Do the following two sentences have the same meaning? Reply with 'equivalent' or 'not_equivalent'.""", """Determine if the following pair of sentences are equivalent. Answer with 'equivalent' or 'not_equivalent'.""", """Assess whether the given two sentences are equivalent. Respond using 'equivalent' or 'not_equivalent'."""

Sentence: """{prompt}"""
Answer:
''')


ENSEMBLE_REWRITING_PROMPT = inspect.cleandoc('''
Given a sentence which could possibly be perturbed through an adversarial attack, respond with the {ensemble_count} diverse and varied candidates for the unperturbed sentence. Do not modify the following words: """{excluded_words}""". Do not answer with anything other than the unperturbed sentences.

Sentence: """Uncovering whether the made coupling of condemns revealed entailment, neutral, or contradiction. Cope with 'entailment', 'neutral', or 'contradiction':"""
Answer: """Identify whether the given pair of sentences demonstrates entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':"""
"""Determine if the given pair of sentences reveals entailment, neutral, or contradiction. Respond with 'entailment', 'neutral', or 'contradiction':"""
"""Decide whether the provided sentence pair displays entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':"""
"""Assess if the presented sentence combination shows entailment, neutral, or contradiction. Reply with 'entailment', 'neutral', or 'contradiction':"""
"""Evaluate whether the given sentence pairing indicates entailment, neutral, or contradiction. Respond using 'entailment', 'neutral', or 'contradiction':"""

Sentence: """Examine the coupling of condemnation and ascertain if their meanings are identical by responses with 'equivalent' or 'not_equivalent'."""
Answer:
"""Examine the pair of sentences and decide if their meanings are identical by answering with 'equivalent' or 'not_equivalent'."""
"""Analyze the provided sentence pair and decide if their meanings are the same by answering with 'equivalent' or 'not_equivalent'."""
"""Assess the relationship between the two sentences and identify if their meanings match by replying with 'equivalent' or 'not_equivalent'."""
"""Evaluate the connection between the given sentences and determine if they convey the same meaning by responding with 'equivalent' or 'not_equivalent'."""
"""Compare the meanings of the presented sentence pair and indicate if they are identical by answering with 'equivalent' or 'not_equivalent'."""

Sentence: """Is the following two trouble equivalent or not? Meet me with 'equivalent' or 'not_equivalent'."""
Answer:
"""Are the following two sentences equivalent or not? Answer with 'equivalent' or 'not_equivalent'."""
"""Are the following two statements equivalent or not? Respond with 'equivalent' or 'not_equivalent'."""
"""Do the following two sentences have the same meaning? Reply with 'equivalent' or 'not_equivalent'."""
"""Determine if the following pair of sentences are equivalent. Answer with 'equivalent' or 'not_equivalent'."""
"""Assess whether the given two sentences are equivalent. Respond using 'equivalent' or 'not_equivalent'."""

Sentence: """{prompt}"""
Answer:
''')

ENSEMBLE_PROMPT_REWRITING_AUGMENTATION_CONFIGS = {
    "ensemble_prompt_rewriting-llama3": {
        "class": EnsemblePromptRewritingAugmentation,
        "rewriting_model": "llama-3-8b-instruct",
        "rewriting_prompt": ENSEMBLE_REWRITING_PROMPT_SAME_LINE,
        "ensemble_count": 5,
    }
}
