# Adapted from the PromptBench library (https://github.com/microsoft/promptbench)

import datasets

LABEL_TO_ID = {
    "sst2": {"negative": 0, "positive": 1},
    "mnli": {"entailment": 0, "neutral": 1, "contradiction": 2},
    "mnli_mismatched": {"entailment": 0, "neutral": 1, "contradiction": 2},
    "mnli_matched": {"entailment": 0, "neutral": 1, "contradiction": 2},
    "qqp": {"equivalent": 1, "not_equivalent": 0},
    "qnli": {"entailment": 0, "not_entailment": 1},
    "rte": {"entailment": 0, "not_entailment": 1},
    "cola": {"unacceptable": 0, "acceptable": 1},
    "mrpc": {"equivalent": 1, "not_equivalent": 0},
    "wnli": {"entailment": 1, "not_entailment": 0},
}

ID_TO_LABEL = {
    "sst2": {0: "negative", 1: "positive"},
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "mnli_matched": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "mnli_mismatched": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "qqp": {1: "equivalent", 0: "not_equivalent"},
    "qnli": {0: "entailment", 1: "not_entailment"},
    "rte": {0: "entailment", 1: "not_entailment"},
    "cola": {0: "unacceptable", 1: "acceptable"},
    "mrpc": {1: "equivalent", 0: "not_equivalent"},
    "wnli": {1: "entailment", 0: "not_entailment"},
}


class Dataset:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.data = self.load_data()

    def get_original_name(self) -> str:
        if "-" not in self.dataset_name:
            return self.dataset_name
        return self.dataset_name.split("-")[0]

    def get_size(self) -> int:
        if "-" not in self.dataset_name:
            return -1
        return int(self.dataset_name.split("-")[1])

    def load_data(self) -> list[dict[str, str | int]]:
        name = self.get_original_name()
        if name not in ["sst2", "cola", "qqp", "mnli", "mnli_matched", "mnli_mismatched", "qnli", "wnli", "rte", "mrpc"]:  # fmt: off
            raise ValueError(f"Dataset {name} is not supported")

        if name == "mnli":
            matched = datasets.load_dataset("glue", "mnli")["validation_matched"]  # type: ignore
            mismatched = datasets.load_dataset("glue", "mnli")["validation_mismatched"]  # type: ignore
            dataset = datasets.concatenate_datasets([matched, mismatched])  # type: ignore
        else:
            dataset = datasets.load_dataset("glue", name)["validation"]  # type: ignore

        data = []

        for sample in dataset:
            if name == "qqp":
                content = "Question 1: " + sample["question1"] + " Question 2: " + sample["question2"]  # type: ignore
            elif name == "qnli":
                content = "Question: " + sample["question"] + " Context: " + sample["sentence"]  # type: ignore
            elif name in ["sst2", "cola"]:
                content = "Sentence: " + sample["sentence"]  # type: ignore
            elif name in ["rte", "mrpc", "wnli"]:
                content = "Sentence 1: " + sample["sentence1"] + " Sentence 2: " + sample["sentence2"]  # type: ignore
            elif name in ["mnli", "mnli_matched", "mnli_mismatched"]:
                content = "Premise: " + sample["premise"] + " Hypothesis: " + sample["hypothesis"]  # type: ignore

            data.append({"content": content, "label": sample["label"]})  # type: ignore

        count = self.get_size()

        if count == -1 or count >= len(data):
            return data

        return data[:count]

    def get_proj_dict(self) -> dict[str, int]:
        return LABEL_TO_ID[self.get_original_name()]

    def get_inv_proj_dict(self) -> dict[int, str]:
        return ID_TO_LABEL[self.get_original_name()]

    def get_class_labels(self) -> list[str]:
        proj_dict = self.get_proj_dict()
        return sorted(proj_dict, key=proj_dict.get)  # type: ignore

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, str | int]:
        return self.data[idx]

    @staticmethod
    def get_class_labels_by_name(dataset_name) -> list[str]:
        proj_dict = LABEL_TO_ID[dataset_name]
        return sorted(proj_dict, key=proj_dict.get)  # type: ignore
