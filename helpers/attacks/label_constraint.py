# Adapted from the PromptBench library (https://github.com/microsoft/promptbench)

import textattack
from textattack.shared import AttackedText


class LabelConstraint(textattack.constraints.PreTransformationConstraint):
    def __init__(self, labels: list[str]):
        self.labels = [label.lower() for label in labels]

    def _get_modifiable_indices(self, current_text: AttackedText):  # type: ignore
        modifiable_indices = set()

        for i, word in enumerate(current_text.words):
            if str(word).lower() not in self.labels:
                modifiable_indices.add(i)

        return modifiable_indices
