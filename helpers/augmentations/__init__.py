import copy
import logging
import pprint

from helpers.utils import LOGGER_NAME, TRACE_LEVEL

from .augmentation import Augmentation, InputFormatter
from .masking.masked_llm_augmentation import MASKED_LLM_AUGMENTATION_CONFIGS
from .masking.masked_lm_augmentation import MASKED_LM_AUGMENTATION_CONFIGS
from .rewriting.ensemble_prompt_rewriting_augmentation import ENSEMBLE_PROMPT_REWRITING_AUGMENTATION_CONFIGS
from .rewriting.iterative_prompt_rewriting_augmentation import ITERATIVE_PROMPT_REWRITING_AUGMENTATION_CONFIGS
from .rewriting.iterative_sampling_prompt_rewriting_augmentation import (
    ITERATIVE_SAMPLING_PROMPT_REWRITING_AUGMENTATION_CONFIGS,
)
from .rewriting.prompt_rewriting_augmentation import PROMPT_REWRITING_AUGMENTATION_CONFIGS
from .rewriting.sampling_prompt_rewriting_augmentation import SAMPLING_PROMPT_REWRITING_AUGMENTATION_CONFIGS
from .synonym.contextual_synonym_augmentation import CONTEXTUAL_SYNONYM_AUGMENTATION_CONFIGS
from .synonym.embedding_synonym_augmentation import EMBEDDING_SYNONYM_AUGMENTATION_CONFIGS
from .synonym.reverse_attack_augmentation import REVERSE_ATTACK_AUGMENTATION_CONFIGS
from .vanilla_augmentation import VANILLA_AUGMENTATION_CONFIGS

logger = logging.getLogger(LOGGER_NAME)

AUGMENTATION_CONFIGS = (
    MASKED_LLM_AUGMENTATION_CONFIGS
    | MASKED_LM_AUGMENTATION_CONFIGS
    | ENSEMBLE_PROMPT_REWRITING_AUGMENTATION_CONFIGS
    | SAMPLING_PROMPT_REWRITING_AUGMENTATION_CONFIGS
    | PROMPT_REWRITING_AUGMENTATION_CONFIGS
    | ITERATIVE_PROMPT_REWRITING_AUGMENTATION_CONFIGS
    | ITERATIVE_SAMPLING_PROMPT_REWRITING_AUGMENTATION_CONFIGS
    | CONTEXTUAL_SYNONYM_AUGMENTATION_CONFIGS
    | EMBEDDING_SYNONYM_AUGMENTATION_CONFIGS
    | REVERSE_ATTACK_AUGMENTATION_CONFIGS
    | VANILLA_AUGMENTATION_CONFIGS
)


def defaults(config):
    return {
        "input_formatter": None,
        "output_formatter": "first_token",
        "generation_mode": "constrained",
        "max_new_tokens": 5,
        "temperature": 0.0,
        "system_prompt": None,
    } | config


def create_augmentation(model_name: str, model_augmentation: str, proj_dict: dict[str, int]) -> Augmentation:
    logger.log(TRACE_LEVEL, f"Augmentations:\n{pprint.pformat(AUGMENTATION_CONFIGS)}")

    if model_augmentation.startswith("siamese"):
        augmentation_config = copy.deepcopy(AUGMENTATION_CONFIGS["vanilla"])
        augmentation_config["siamese_sft_config"] = model_augmentation.removeprefix("siamese-")
    else:
        augmentation_config = copy.deepcopy(AUGMENTATION_CONFIGS[model_augmentation])

    augmentation_class = augmentation_config.pop("class")

    return augmentation_class(
        model_name=model_name,
        model_augmentation=model_augmentation,
        proj_dict=copy.deepcopy(proj_dict),
        **defaults(augmentation_config),
    )


def get_sft_config():
    return defaults({"generation_mode": "freeform"})


__all__ = ["create_augmentation", "InputFormatter"]
