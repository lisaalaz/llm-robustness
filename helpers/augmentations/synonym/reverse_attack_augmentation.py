import logging

import lru
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.goal_function_results import ClassificationGoalFunctionResult
from textattack.goal_functions import GoalFunction
from textattack.search_methods import (
    AlzantotGeneticAlgorithm,
    BeamSearch,
    GreedySearch,
    GreedyWordSwapWIR,
    ImprovedGeneticAlgorithm,
    ParticleSwarmOptimization,
)
from textattack.shared import AttackedText
from textattack.transformations import WordSwapEmbedding

from helpers.attacks import Attack, LabelConstraint
from helpers.augmentations import Augmentation
from helpers.utils import LOGGER_NAME, perplexity

logger = logging.getLogger(LOGGER_NAME)


class ReverseAttackAugmentation(Augmentation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # kwargs
        self.perplexity_model: str = kwargs.pop("perplexity_model")
        self.iterations: int = kwargs.pop("iterations")

        # init
        self.label_constraint = self.class_labels + ["[UNK]"] + [f"{label}'" for label in self.class_labels]
        self.compute_perplexity = perplexity(self.perplexity_model)
        self.search_method = self.get_search_method(kwargs.pop("search_method"))

        self.print_augmentation_config()

    def get_search_method(self, search_method):
        if search_method == "greedy_search":
            return GreedySearch()
        if search_method == "greedy_word_swap_wir":
            return GreedyWordSwapWIR()
        elif search_method == "beam_search":
            return BeamSearch()
        elif search_method == "alzantot_genetic_algorithm":
            return AlzantotGeneticAlgorithm()
        elif search_method == "improved_genetic_algorithm":
            return ImprovedGeneticAlgorithm()
        elif search_method == "partical_swarm_optimization":
            return ParticleSwarmOptimization()
        else:
            raise ValueError(f"Invalid search method: {search_method}")

    def augmentation(self, prompt: str) -> str | list[str]:
        transformation = WordSwapEmbedding()
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        constraints.append(LabelConstraint(self.label_constraint))
        goal_function = ReconstructedPromptGoalFunction(self.compute_perplexity, self.iterations)

        prompt_perplexity = self.compute_perplexity(predictions=[prompt])[0]
        logger.debug(f"Original prompt perplexity: {prompt_perplexity:.4f}")

        result = Attack(goal_function, constraints, transformation, self.search_method).run(prompt, prompt_perplexity)

        new_prompt = result.perturbed_result.attacked_text.text
        new_prompt_perplexity = self.compute_perplexity(predictions=[new_prompt])[0]
        logger.debug(f"Augmented prompt perplexity: {new_prompt_perplexity:.4f}")

        if prompt_perplexity < new_prompt_perplexity:
            return prompt

        return new_prompt


class ReconstructedPromptGoalFunction(GoalFunction):
    def __init__(self, compute_perplexity, query_budget):
        self.compute_perplexity = compute_perplexity
        self.query_budget = query_budget

        self.maximizable = True
        self.use_cache = True
        self.batch_size = 32

        self._call_model_cache = lru.LRU(2**20)

    def _call_model_uncached(self, attacked_text_list: list[AttackedText]):
        return [self.compute_perplexity(predictions=[prompt.text]) for prompt in attacked_text_list]

    def _is_goal_complete(self, model_output, attacked_text):
        return False

    def _get_score(self, model_output, attacked_text):
        return -model_output[0]

    def _goal_function_result_type(self):
        return ClassificationGoalFunctionResult

    def _process_model_outputs(self, inputs, outputs):
        return outputs


REVERSE_ATTACK_AUGMENTATION_CONFIGS = {
    "reverse_attack-greedy_search": {
        "class": ReverseAttackAugmentation,
        "search_method": "greedy_search",
        "perplexity_model": "gpt2",
        "iterations": 10,
    },
    "reverse_attack-beam_search": {
        "class": ReverseAttackAugmentation,
        "search_method": "beam_search",
        "perplexity_model": "gpt2",
        "iterations": 10,
    },
}
