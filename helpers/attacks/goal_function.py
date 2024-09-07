# Adapted from the PromptBench library (https://github.com/microsoft/promptbench)

import lru
import textattack
from textattack.goal_function_results import ClassificationGoalFunctionResult
from textattack.shared import AttackedText

from helpers.utils import calculate_pdr


class GoalFunction(textattack.goal_functions.GoalFunction):
    def __init__(self, model, dataset, eval_func, query_budget):
        self.model = model
        self.dataset = dataset
        self.eval_func = eval_func
        self.query_budget = query_budget

        self.maximizable = True
        self.use_cache = True
        self.batch_size = 32

        if self.use_cache:
            self._call_model_cache = lru.LRU(2**20)
        else:
            self._call_model_cache = None

    def _call_model_uncached(self, attacked_text_list: list[AttackedText]):
        return [self.eval_func(prompt.text, self.dataset, self.model) for prompt in attacked_text_list]

    def _is_goal_complete(self, model_output, attacked_text):
        return False

    def _get_score(self, model_output, attacked_text):
        return calculate_pdr(self.ground_truth_output, model_output)

    def _goal_function_result_type(self):
        return ClassificationGoalFunctionResult

    def _process_model_outputs(self, inputs, outputs):
        return outputs
