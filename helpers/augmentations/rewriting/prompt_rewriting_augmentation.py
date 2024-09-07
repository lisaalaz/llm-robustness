import inspect
import logging

from helpers.augmentations import Augmentation
from helpers.models import create_model
from helpers.utils import LOGGER_NAME, perplexity

logger = logging.getLogger(LOGGER_NAME)

ESC = '"""'
MAX_PROMPT_LENGTH = 512


class PromptRewritingAugmentation(Augmentation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # kwargs
        self.rewriting_model_name: str = kwargs.pop("rewriting_model")
        self.rewriting_prompt: str = kwargs.pop("rewriting_prompt")
        self.perplexity_model: str = kwargs.pop("perplexity_model")
        self.minimize_perplexity: bool = kwargs.pop("minimize_perplexity")

        # init
        self.compute_perplexity = perplexity(self.perplexity_model)
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
        prompt_perplexity = self.compute_perplexity(predictions=[prompt])[0]
        logger.debug(f"Original prompt perplexity: {prompt_perplexity:.4f}")

        rewriting_prompt = self.rewriting_prompt.replace(
            "{excluded_words}", str(self.class_labels + ["[UNK]"])
        ).replace("{prompt}", prompt)
        output = self.rewriting_model.predict(rewriting_prompt, **self.model_kwargs)
        rewritten_prompt = self.remove_prefix_suffix(output.split("\n")[-1])
        if not rewritten_prompt:
            logger.debug("Rewritten prompt is empty, using original prompt")
            return prompt
        logger.debug(f"Rewritten prompt: {repr(rewritten_prompt)}")

        for label in self.class_labels:
            formatted_label = f"'{label}'"
            if formatted_label not in rewritten_prompt:
                logger.debug(f"Rewritten prompt doesn't contain label {formatted_label}, using original prompt")
                return prompt

        if len(rewritten_prompt) > MAX_PROMPT_LENGTH:
            logger.debug("Rewritten prompt is too long, using original prompt")
            return prompt

        rewritten_prompt_perplexity = self.compute_perplexity(predictions=[rewritten_prompt])[0]
        logger.debug(f"Rewritten prompt perplexity: {rewritten_prompt_perplexity:.4f}")
        if self.minimize_perplexity and rewritten_prompt_perplexity > prompt_perplexity:
            logger.debug("Rewritten prompt has higher perplexity, using original prompt")
            return prompt
        return rewritten_prompt

    def remove_prefix_suffix(self, string: str):
        return string.strip("\"' ")

    def clear_memory(self):
        self.rewriting_model.clear_memory()
        return super().clear_memory()


REWRITING_PROMPT_ORIGINAL = inspect.cleandoc('''
Given a sentence which could be perturbed through an adversarial attack, respond with the unperturbed sentence. Do not modify the following words: """{excluded_words}""".Do not answer with anything other than the unperturbed sentence.

Here are three examples:

Example sentence 0: """Uncovering whether the made coupling of condemns revealed entailment, neutral, or contradiction. Cope with 'entailment', 'neutral', or 'contradiction':"""
Example answer 0: Identify whether the given pair of sentences demonstrates entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':

Example sentence 1: """Specifies if the made coupling of condemns exposure entailment, neutral, or contradiction. Reacting with 'entailment', 'neutral', or 'contradiction':"""
Example answer 1: Determine if the given pair of sentences displays entailment, neutral, or contradiction. Respond with 'entailment', 'neutral', or 'contradiction':

Example sentence 2: """Ca the ratio between the offered penalty be entailment, neutral, or contradiction? Reactions with 'entailment', 'neutral', or 'contradiction':"""
Example answer 2: Does the relationship between the given sentences represent entailment, neutral, or contradiction? Respond with 'entailment', 'neutral', or 'contradiction':

Here is the real sentence:

Sentence: """{prompt}"""
Answer:
''')

REWRITING_PROMPT_FIXED = inspect.cleandoc('''
Given a sentence which could be perturbed through an adversarial attack, respond with the unperturbed sentence. Do not modify the following words: {excluded_words}. Do not answer with anything other than the unperturbed sentence.

Here are three examples:

Example sentence 1: """Uncovering whether the made coupling of condemns revealed entailment, neutral, or contradiction. Cope with 'entailment', 'neutral', or 'contradiction':"""
Example answer 1: """Identify whether the given pair of sentences demonstrates entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':"""

Example sentence 2: """Specifies if the made coupling of condemns exposure entailment, neutral, or contradiction. Reacting with 'entailment', 'neutral', or 'contradiction':"""
Example answer 2: """Determine if the given pair of sentences displays entailment, neutral, or contradiction. Respond with 'entailment', 'neutral', or 'contradiction':"""

Example sentence 3: """Ca the ratio between the offered penalty be entailment, neutral, or contradiction? Reactions with 'entailment', 'neutral', or 'contradiction':"""
Example answer 3: """Does the relationship between the given sentences represent entailment, neutral, or contradiction? Respond with 'entailment', 'neutral', or 'contradiction':"""

Here is the real sentence:

Sentence: """{prompt}"""
Answer:
''')

REWRITING_PROMPT_NO_FS_CONTEXT = inspect.cleandoc('''
Given a sentence which could be perturbed through an adversarial attack, respond with the unperturbed sentence. Do not modify the following words: {excluded_words}. Do not answer with anything other than the unperturbed sentence.

Sentence: """Uncovering whether the made coupling of condemns revealed entailment, neutral, or contradiction. Cope with 'entailment', 'neutral', or 'contradiction':"""
Answer: """Identify whether the given pair of sentences demonstrates entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':"""

Sentence: """Specifies if the made coupling of condemns exposure entailment, neutral, or contradiction. Reacting with 'entailment', 'neutral', or 'contradiction':"""
Answer: """Determine if the given pair of sentences displays entailment, neutral, or contradiction. Respond with 'entailment', 'neutral', or 'contradiction':"""

Sentence: """Ca the ratio between the offered penalty be entailment, neutral, or contradiction? Reactions with 'entailment', 'neutral', or 'contradiction':"""
Answer: """Does the relationship between the given sentences represent entailment, neutral, or contradiction? Respond with 'entailment', 'neutral', or 'contradiction':"""

Sentence: """{prompt}"""
Answer:
''')

REWRITING_PROMPT_VARIETY = inspect.cleandoc('''
Given a sentence which could be perturbed through an adversarial attack, respond with the unperturbed sentence. Do not modify the following words: {excluded_words}. Do not answer with anything other than the unperturbed sentence.

Sentence: """Uncovering whether the made coupling of condemns revealed entailment, neutral, or contradiction. Cope with 'entailment', 'neutral', or 'contradiction':"""
Answer: """Identify whether the given pair of sentences demonstrates entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':"""

Sentence: """Examine the coupling of condemnation and ascertain if their meanings are identical by responses with 'equivalent' or 'not_equivalent'."""
Answer: """Examine the pair of sentences and decide if their meanings are identical by answering with 'equivalent' or 'not_equivalent'."""

Sentence: """Is the following two trouble equivalent or not? Meet me with 'equivalent' or 'not_equivalent'."""
Answer: """Are the following two sentences equivalent or not? Answer with 'equivalent' or 'not_equivalent'."""

Sentence: """{prompt}"""
Answer:
''')

REWRITING_PROMPT_VARIETY_GENERAL = inspect.cleandoc('''
Given a instruction that may contain errors, alterations, or suboptimal language use, provide an improved version that accurately conveys the intended meaning of the instruction. Do not modify the following words: """{excluded_words}""". Do not answer with anything other than the new instruction.

Sentence: """Uncovering whether the made coupling of condemns revealed entailment, neutral, or contradiction. Cope with 'entailment', 'neutral', or 'contradiction':"""
Answer: """Identify whether the given pair of sentences demonstrates entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':"""

Sentence: """Examine the coupling of condemnation and ascertain if their meanings are identical by responses with 'equivalent' or 'not_equivalent'."""
Answer: """Examine the pair of sentences and decide if their meanings are identical by answering with 'equivalent' or 'not_equivalent'."""

Sentence: """Is the following two trouble equivalent or not? Meet me with 'equivalent' or 'not_equivalent'."""
Answer: """Are the following two sentences equivalent or not? Answer with 'equivalent' or 'not_equivalent'."""

Sentence: """{prompt}"""
Answer:
''')

REWRITING_PROMPT_GENERAL = inspect.cleandoc('''
Given a instruction that may contain errors, alterations, or suboptimal language use, provide an improved version that accurately conveys the intended meaning of the instruction. Do not modify the following words: """{excluded_words}""". Do not answer with anything other than the new instruction.

Sentence: """Uncovering whether the made coupling of condemns revealed entailment, neutral, or contradiction. Cope with 'entailment', 'neutral', or 'contradiction':"""
Answer: """Identify whether the given pair of sentences demonstrates entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':"""

Sentence: """Specifies if the made coupling of condemns exposure entailment, neutral, or contradiction. Reacting with 'entailment', 'neutral', or 'contradiction':"""
Answer: """Determine if the given pair of sentences displays entailment, neutral, or contradiction. Respond with 'entailment', 'neutral', or 'contradiction':"""

Sentence: """Ca the ratio between the offered penalty be entailment, neutral, or contradiction? Reactions with 'entailment', 'neutral', or 'contradiction':"""
Answer: """Does the relationship between the given sentences represent entailment, neutral, or contradiction? Respond with 'entailment', 'neutral', or 'contradiction':"""

Sentence: """{prompt}"""
Answer:
''')

PROMPT_REWRITING_AUGMENTATION_CONFIGS = {
    "prompt_rewriting-llama3": {
        "class": PromptRewritingAugmentation,
        "rewriting_model": "llama-3-8b-instruct",
        "rewriting_prompt": REWRITING_PROMPT_ORIGINAL,
        "perplexity_model": "gpt2",
        "minimize_perplexity": True,
    },
    "prompt_rewriting-llama3-no_min_perplexity": {
        "class": PromptRewritingAugmentation,
        "rewriting_model": "llama-3-8b-instruct",
        "rewriting_prompt": REWRITING_PROMPT_ORIGINAL,
        "perplexity_model": "gpt2",
        "minimize_perplexity": False,
    },
    "prompt_rewriting-llama3-fixed": {
        "class": PromptRewritingAugmentation,
        "rewriting_model": "llama-3-8b-instruct",
        "rewriting_prompt": REWRITING_PROMPT_FIXED,
        "perplexity_model": "gpt2",
        "minimize_perplexity": True,
    },
    "prompt_rewriting-llama3-no_fs_context": {
        "class": PromptRewritingAugmentation,
        "rewriting_model": "llama-3-8b-instruct",
        "rewriting_prompt": REWRITING_PROMPT_NO_FS_CONTEXT,
        "perplexity_model": "gpt2",
        "minimize_perplexity": True,
    },
    "prompt_rewriting-llama3-variety": {
        "class": PromptRewritingAugmentation,
        "rewriting_model": "llama-3-8b-instruct",
        "rewriting_prompt": REWRITING_PROMPT_VARIETY,
        "perplexity_model": "gpt2",
        "minimize_perplexity": True,
    },
    "prompt_rewriting-llama3-general": {
        "class": PromptRewritingAugmentation,
        "rewriting_model": "llama-3-8b-instruct",
        "rewriting_prompt": REWRITING_PROMPT_GENERAL,
        "perplexity_model": "gpt2",
        "minimize_perplexity": True,
    },
    "prompt_rewriting-llama3-variety-general": {
        "class": PromptRewritingAugmentation,
        "rewriting_model": "llama-3-8b-instruct",
        "rewriting_prompt": REWRITING_PROMPT_VARIETY_GENERAL,
        "perplexity_model": "gpt2",
        "minimize_perplexity": True,
    },
    "prompt_rewriting-llama3-sft-zs-promptbench-attackaware": {
        "class": PromptRewritingAugmentation,
        "rewriting_model": "aryanagrawal1/llama-3-8b-instruct-sft-rewriting-zs-promptbench-attackaware",
        "rewriting_prompt": REWRITING_PROMPT_FIXED,
        "perplexity_model": "gpt2",
        "minimize_perplexity": False,
    },
    "prompt_rewriting-llama3-sft-fs-promptbench-attackaware": {
        "class": PromptRewritingAugmentation,
        "rewriting_model": "aryanagrawal1/llama-3-8b-instruct-sft-rewriting-fs-promptbench-attackaware",
        "rewriting_prompt": REWRITING_PROMPT_FIXED,
        "perplexity_model": "gpt2",
        "minimize_perplexity": False,
    },
}
