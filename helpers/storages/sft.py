import inspect
import json
import logging
import os
from collections import defaultdict

from datasets import Dataset as HuggingFaceDataset
from datasets import concatenate_datasets

from helpers.utils import LOGGER_NAME

from .dataset import Dataset
from .prompt_sets import SFT_PROMPTS

logger = logging.getLogger(LOGGER_NAME)

# fmt: off
SFT_REWRITING_PROMPT_TEMPLATES = {
"zeroshot": inspect.cleandoc('''
Given a sentence which could be perturbed through an adversarial attack, respond with the unperturbed sentence. Do not modify the following words: {excluded_words}. Do not answer with anything other than the unperturbed sentence.

Sentence: """{sentence}"""
Answer:
'''),

"fewshot": inspect.cleandoc('''
Given a sentence which could be perturbed through an adversarial attack, respond with the unperturbed sentence. Do not modify the following words: {excluded_words}. Do not answer with anything other than the unperturbed sentence.

Here are three examples:

Example sentence 1: """Uncovering whether the made coupling of condemns revealed entailment, neutral, or contradiction. Cope with 'entailment', 'neutral', or 'contradiction':"""
Example answer 1: """Identify whether the given pair of sentences demonstrates entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':"""

Example sentence 2: """Specifies if the made coupling of condemns exposure entailment, neutral, or contradiction. Reacting with 'entailment', 'neutral', or 'contradiction':"""
Example answer 2: """Determine if the given pair of sentences displays entailment, neutral, or contradiction. Respond with 'entailment', 'neutral', or 'contradiction':"""

Example sentence 3: """Ca the ratio between the offered penalty be entailment, neutral, or contradiction? Reactions with 'entailment', 'neutral', or 'contradiction':"""
Example answer 3: """Does the relationship between the given sentences represent entailment, neutral, or contradiction? Respond with 'entailment', 'neutral', or 'contradiction':"""

Here is the real sentence:

Sentence: """{sentence}"""
Answer:
''')
}
# fmt: on


class AdvGLUE:
    splits = ["dev", "test"]
    dataset_name_map = {"qqp": "qqp", "mnli": "mnli_matched", "mnli-mm": "mnli_mismatched", "rte": "rte"}
    data_path = f"{os.environ['SFT_DATA']}/advglue"

    @staticmethod
    def download_advglue():
        for split in AdvGLUE.splits:
            if os.path.exists(f"{AdvGLUE.data_path}/{split}.json"):
                logger.info(f"AdvGLUE data for {split} split already downloaded, continuing...")
                continue

            run_file = os.path.expanduser(f"{AdvGLUE.data_path}/download_advglue.sh")
            with open(run_file, "w") as file:
                download_script = f"wget -O {AdvGLUE.data_path}/{split}.json https://adversarialglue.github.io/dataset/{split}_ann.json"  # fmt: skip
                file.writelines(["#!/bin/bash", download_script])

            os.makedirs(AdvGLUE.data_path, exist_ok=True)
            os.system(f"chmod +x {run_file}")
            os.system(run_file)

    @staticmethod
    def process_dataset() -> dict[str, list[dict[str, str]]]:
        files = [f"{AdvGLUE.data_path}/{split}.json" for split in AdvGLUE.splits]
        data = defaultdict(list)

        for file in files:
            with open(file) as f:
                advglue = json.load(f)

            for key in ["sst2", "qnli"]:
                del advglue[key]

            for dataset, samples in advglue.items():
                new_dataset = []
                for sample in samples:
                    if sample["method"].lower() in ["glue"] or not any("original" in key for key in sample.keys()):
                        continue

                    new_sample = {}
                    for key in sample.keys():
                        if "original" in key:
                            new_sample["original"] = sample[key]
                            new_sample["attacked"] = sample[key.removeprefix("original_")].strip()
                            break

                    new_dataset.append(new_sample)

                data[dataset].extend(new_dataset)

        return data

    @staticmethod
    def get_rewriting_dataset(sft_config: dict) -> HuggingFaceDataset:
        rewriting_template = sft_config["rewriting_template"]
        dataset = []

        for dataset_name, sentence_pairs in AdvGLUE.process_dataset().items():
            for pair in sentence_pairs:
                formatted_prompt = (
                    SFT_REWRITING_PROMPT_TEMPLATES[rewriting_template]
                    .replace(
                        "{excluded_words}",
                        str(Dataset.get_class_labels_by_name(AdvGLUE.dataset_name_map[dataset_name])),
                    )
                    .replace("{sentence}", pair["attacked"])
                )
                formatted_completion = f'"""{pair["original"]}"""'
                dataset.append({"prompt": formatted_prompt, "completion": formatted_completion})

        return HuggingFaceDataset.from_list(dataset)

    @staticmethod
    def get_siamese_dataset() -> HuggingFaceDataset:
        dataset = []

        for sentence_pairs in AdvGLUE.process_dataset().values():
            dataset.extend(sentence_pairs)

        return HuggingFaceDataset.from_list(dataset)

    @staticmethod
    def get_dataset(task, sft_config):
        AdvGLUE.download_advglue()
        if task == "siamese":
            return AdvGLUE.get_siamese_dataset()
        elif task == "rewriting":
            return AdvGLUE.get_rewriting_dataset(sft_config)
        else:
            raise ValueError(f"Unknown task {task}")


class PromptBench:
    @staticmethod
    def get_prompts(dataset_version: str, dataset_name: str, prompt_count: int = -1) -> list[dict[str, str]]:
        prompts = []

        for prompt_list in SFT_PROMPTS[dataset_version][dataset_name]:
            original = prompt_list[0].decode("utf-8") if isinstance(prompt_list[0], bytes) else prompt_list[0]
            for p in prompt_list[1:]:
                attacked = p.decode("utf-8") if isinstance(p, bytes) else p
                prompts.append({"attacked": attacked, "original": original})

        if prompt_count > len(prompts) or prompt_count == -1:
            x = f"Using all {len(prompts)} prompts for {dataset_name}"
            if prompt_count > len(prompts):
                logger.warning(f"{x} (requested {prompt_count})")
            else:
                logger.info(x)
            return prompts

        return prompts[:prompt_count]

    @staticmethod
    def get_rewriting_dataset(sft_config: dict) -> HuggingFaceDataset:
        dataset_version = sft_config["dataset_version"]
        rewriting_template = sft_config["rewriting_template"]
        dataset = []

        for dataset_name in SFT_PROMPTS[dataset_version]:
            prompts = PromptBench.get_prompts(dataset_version, dataset_name)

            for prompt_pair in prompts:
                formatted_prompt = (
                    SFT_REWRITING_PROMPT_TEMPLATES[rewriting_template]
                    .replace("{excluded_words}", str(Dataset.get_class_labels_by_name(dataset_name)))
                    .replace("{sentence}", prompt_pair["attacked"])
                )
                formatted_completion = f'"""{prompt_pair["original"]}"""'
                dataset.append({"prompt": formatted_prompt, "completion": formatted_completion})

        return HuggingFaceDataset.from_list(dataset)

    @staticmethod
    def get_siamese_dataset(sft_config: dict) -> HuggingFaceDataset:
        dataset_version = sft_config["dataset_version"]
        dataset = []

        for dataset_name in SFT_PROMPTS[dataset_version]:
            prompts = PromptBench.get_prompts(dataset_version, dataset_name)
            dataset.extend(prompts)

        return HuggingFaceDataset.from_list(dataset)

    @staticmethod
    def get_dataset(task, sft_config):
        if task == "siamese":
            return PromptBench.get_siamese_dataset(sft_config)
        elif task == "rewriting":
            return PromptBench.get_rewriting_dataset(sft_config)
        else:
            raise ValueError(f"Unknown task {task}")


def get_sft_dataset(task: str, sft_config: dict) -> HuggingFaceDataset:
    dataset_version = sft_config["dataset_version"]

    if dataset_version.startswith("promptbench"):
        return PromptBench.get_dataset(task, sft_config)
    elif dataset_version == "advglue":
        return AdvGLUE.get_dataset(task, sft_config)
    elif dataset_version == "advglue-promptbench":
        sft_config["dataset_version"] = "advglue"
        advglue = AdvGLUE.get_dataset(task, sft_config)
        sft_config["dataset_version"] = "promptbench"
        promptbench = PromptBench.get_dataset(task, sft_config)
        return concatenate_datasets([advglue, promptbench]).shuffle()

    raise ValueError(f"Unknown dataset version {dataset_version}")
