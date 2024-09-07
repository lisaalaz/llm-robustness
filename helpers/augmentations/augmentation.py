import collections
import logging
import pprint
from abc import ABC
from functools import cache

from spellchecker import SpellChecker

from helpers.models import EXCLUDED_LOG_KEYS, create_model
from helpers.utils import LOGGER_NAME, TRACE_LEVEL, InputFormatter, OutputFormatter

logger = logging.getLogger(LOGGER_NAME)


class Augmentation(ABC):
    def __init__(self, **kwargs):
        # kwargs
        self.model_name: str = kwargs.pop("model_name")
        self.model_augmentation: str = kwargs.pop("model_augmentation")
        self.proj_dict: dict[str, int] = kwargs.pop("proj_dict")
        self.input_formatter: str = kwargs.pop("input_formatter")
        self.output_formatter: str = kwargs.pop("output_formatter")

        # init
        self.class_labels = list(self.proj_dict.keys())
        self.excluded_words = set([f"'{label}" for label in self.class_labels] + self.class_labels + ["[UNK]"])

        self.format_input = InputFormatter.format_input
        self.format_output = OutputFormatter(self.output_formatter, self.proj_dict, self.class_labels).format

        if self.input_formatter == "spellcheck":
            self.spell_checker = SpellChecker()

        self.model = create_model(self.model_name, model_context="prediction", output_words=self.class_labels, **kwargs)

        if kwargs.get("adapter_name"):
            self.model.load_adapter(kwargs["adapter_name"])

    def print_augmentation_config(self):
        print_dict = {x: self.__dict__[x] for x in self.__dict__ if x not in EXCLUDED_LOG_KEYS}
        logger.info(f"\nAugmentation config:\n{pprint.pformat(print_dict)}\n")

    def _predict(self, prompt: str, sample: str, **kwargs) -> int:
        input_text = self.format_input(prompt, sample)
        output_text = self.model.predict(input_text, **kwargs)
        return self.format_output(output_text)

    def predict(self, prompt: str | list[str], sample: str, **kwargs) -> int:
        if isinstance(prompt, str):
            return self._predict(prompt, sample, **kwargs)

        preds = [self._predict(p, sample, **kwargs) for p in prompt]
        logger.log(TRACE_LEVEL, f"Ensemble predictions: {preds}")
        return collections.Counter(preds).most_common(1)[0][0]

    def augmentation(self, prompt: str) -> str | list[str]:
        return prompt

    @cache
    def augment_prompt(self, prompt: str) -> str | list[str]:
        if self.input_formatter == "spellcheck":
            prompt = InputFormatter.spellcheck(prompt, self.spell_checker, self.excluded_words)
        return self.augmentation(prompt)

    def clear_memory(self):
        self.model.clear_memory()
