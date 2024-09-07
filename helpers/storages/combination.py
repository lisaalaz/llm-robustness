import pprint


class Combination:
    def __init__(
        self, task: str, model_name: str, model_augmentation: str, dataset_name: str, attack_name: str, prompt_type: str
    ):
        self.task = task
        self.model_name = model_name
        self.model_augmentation = model_augmentation
        self.dataset_name = dataset_name
        self.attack_name = attack_name
        self.prompt_type = prompt_type

    def __str__(self):
        return pprint.pformat({k: v for k, v in self.__dict__.items() if v != "n/a"}, sort_dicts=False)

    def path_name(self) -> str:
        return f"{self.model_name}/{self.dataset_name}/{self.attack_name}/{self.prompt_type}/{self.task}-{self.model_augmentation}.json"
