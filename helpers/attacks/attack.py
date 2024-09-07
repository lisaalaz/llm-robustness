# Adapted from the PromptBench library (https://github.com/microsoft/promptbench)

from collections import OrderedDict

import textattack
from textattack.attack_results import SkippedAttackResult
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.shared import AttackedText


class Attack(textattack.attack.Attack):
    def attack(self, example, ground_truth_output):
        if isinstance(example, (str, OrderedDict)):
            example = AttackedText(example)

        goal_function_result, _ = self.goal_function.init_attack_example(example, ground_truth_output)

        if goal_function_result.goal_status == GoalFunctionResultStatus.SKIPPED:  # type: ignore
            return SkippedAttackResult(goal_function_result)
        else:
            return self._attack(goal_function_result)

    def run(self, example, ground_truth_output):
        return self.attack(example, ground_truth_output)
