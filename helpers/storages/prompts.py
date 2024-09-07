import copy
import logging

from helpers.utils import LOGGER_NAME

from .combination import Combination
from .prompt_sets import ROLE_ORIENTED_PROMPTS, TASK_ORIENTED_PROMPTS
from .storage_interface import StorageInterface

logger = logging.getLogger(LOGGER_NAME)


class Prompts:
    @staticmethod
    def get_eval_prompts(
        dataset_name: str, storage: StorageInterface, combination: Combination, prompt_source: str
    ) -> list[dict]:
        prompt_source_combination = copy.deepcopy(combination)
        prompt_source_combination.task = "attack"

        if not prompt_source.startswith("same"):
            prompt_source_combination.model_augmentation = prompt_source.split("-")[0]

        prompt_source_combination.dataset_name = f"{dataset_name}-{prompt_source.split('-')[1]}"
        prompts = storage.load(prompt_source_combination)

        if prompts is None:
            raise ValueError(f"No prompts found for {prompt_source}, please run attacks first")

        return prompts

    @staticmethod
    def get_attack_prompts(dataset_name: str, prompt_type: str, prompt_count: int):
        if "fewshot" in prompt_type:
            raise ValueError("Fewshot prompts are not supported for attacks")

        if "task" in prompt_type:
            return TASK_ORIENTED_PROMPTS[dataset_name][:prompt_count]
        elif "role" in prompt_type:
            return ROLE_ORIENTED_PROMPTS[dataset_name][:prompt_count]
        else:
            raise ValueError(f"Prompt type {prompt_type} not found")
