# Adapted from the PromptBench library (https://github.com/microsoft/promptbench)

import logging

from helpers.utils import LOGGER_NAME, time_function

from .attack import Attack
from .goal_function import GoalFunction
from .label_constraint import LabelConstraint
from .recipes import AttackRecipes

logger = logging.getLogger(LOGGER_NAME)

LABEL_SET = {
    "sst2": ["positive", "negative", "positive'", "negative'", "0", "1", "0'", "1'"],
    "mnli": ["entailment", "neutral", "contradiction", "entailment'", "neutral'", "contradiction'"],
    "mnli_mismatched": ["entailment", "neutral", "contradiction", "entailment'", "neutral'", "contradiction'"],
    "mnli_matched": ["entailment", "neutral", "contradiction", "entailment'", "neutral'", "contradiction'"],
    "qqp": ["equivalent", "not_equivalent", "equivalent'", "not_equivalent'"],
    "qnli": ["entailment", "not_entailment", "entailment'", "not_entailment'", "0", "1", "0'", "1'"],
    "rte": ["entailment", "not_entailment", "entailment'", "not_entailment'", "0", "1", "0'", "1'"],
    "cola": ["unacceptable", "acceptable", "unacceptable'", "acceptable'"],
    "mrpc": ["equivalent", "not_equivalent", "equivalent'", "not_equivalent'"],
    "wnli": ["entailment", "not_entailment", "entailment'", "not_entailment'", "0", "1", "0'", "1'"],
    "mmlu": ["A", "B", "C", "D", "A'", "B'", "C'", "D'", "a", "b", "c", "d", "a'", "b'", "c'", "d'"],
    "squad_v2": ["unanswerable", "unanswerable'"],
    "iwslt": ["translate", "translate'"],
    "un_multi": ["translate", "translate'"],
    "math": ["math", "math'"],
    "bool_logic": ["True", "False", "True'", "False'", "bool", "boolean", "bool'", "boolean'"],
    "valid_parentheses": [
        "Valid",
        "Invalid",
        "Valid'",
        "Invalid'",
        "matched",
        "matched'",
        "valid",
        "invalid",
        "valid'",
        "invalid'",
    ],
}


class Attacker:
    def __init__(self, model, dataset, attack_name: str, prompt: str, eval_func):
        self.model = model
        self.dataset = dataset
        self.attack_name = attack_name
        self.prompt = prompt
        self.eval_func = eval_func

        self.unmodifiable_words = LABEL_SET[dataset.get_original_name()]
        self.goal_function = GoalFunction(model, dataset, eval_func, query_budget=float("inf"))
        self.prompt_attack = self.create_attack(attack_name)

    def create_attack(self, attack):
        recipe = getattr(AttackRecipes, attack)

        transformation, constraints, search_method = recipe()
        constraints.append(LabelConstraint(self.unmodifiable_words))
        return Attack(self.goal_function, constraints, transformation, search_method)

    @time_function(message="Attack duration")
    def attack(self):
        ground_truth_output = self.eval_func(self.prompt, self.dataset, self.model)
        result = self.prompt_attack.run(self.prompt, ground_truth_output)

        logger.info(f"Number of iterations over the dataset: {result.num_queries}")
        return {
            "original_prompt": result.original_result.attacked_text.text,
            "attacked_prompt": result.perturbed_result.attacked_text.text,
            "original_score": result.original_result.output,
            "attacked_score": result.perturbed_result.output,
        }
