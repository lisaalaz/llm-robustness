import logging
from collections import Counter

from sklearn.metrics import accuracy_score

from helpers.attacks import Attacker
from helpers.augmentations import Augmentation as AugmentedModel
from helpers.augmentations import create_augmentation
from helpers.storages import Dataset, Prompts, StorageInterface
from helpers.utils import (
    LOGGER_NAME,
    TRACE_LEVEL,
    get_combinations,
    log_augmented_prompt,
    log_classification_reports,
    log_results,
    time_function,
)

logger = logging.getLogger(LOGGER_NAME)


def eval(prompt: str, dataset: Dataset, model: AugmentedModel, return_preds: bool = False):
    preds = []
    gts = []

    logger.debug(f"Prompt: {repr(prompt)}")
    augmented_prompt = model.augment_prompt(prompt)
    log_augmented_prompt(augmented_prompt, prompt)

    for sample in dataset:
        content, gt = sample["content"], sample["label"]
        pred = model.predict(augmented_prompt, content)  # type: ignore
        logger.log(TRACE_LEVEL, f"Predicted label: {pred}")
        logger.log(TRACE_LEVEL, f"Ground truth label: {gt}")
        preds.append(pred)
        gts.append(gt)

    preds_count = dict(Counter(preds))
    gts_count = dict(Counter(gts))

    if logger.isEnabledFor(logging.DEBUG):
        for idx, class_label in enumerate(dataset.get_class_labels()):
            preds_count[class_label] = preds_count.get(idx, 0)
            gts_count[class_label] = gts_count.get(idx, 0)
        logger.debug(f"Predicted labels: {preds_count}")
        logger.debug(f"Ground truth labels: {gts_count}")

    logger.debug(f"Unknown predictions: {preds_count.get(-1, 0)}")
    accuracy = accuracy_score(gts, preds)
    logger.debug(f"Accuracy: {accuracy:.4f}")

    if return_preds:
        return accuracy, gts, preds
    return accuracy


@time_function(message="Config duration")
def run_attack(original_prompts: list[str], dataset: Dataset, model: AugmentedModel, attack_name: str) -> list:
    output = []

    for idx, prompt in enumerate(original_prompts):
        logger.info(f"Attacking prompt {idx + 1}/{len(original_prompts)}")
        attacker = Attacker(model, dataset, attack_name, prompt, eval)
        results = attacker.attack()
        log_results(results)
        output.append(results)

    return output


@time_function(message="Config duration")
def run_evaluation(prompts: list[dict], dataset: Dataset, model: AugmentedModel) -> list:
    raw_data = {}

    for idx, pair in enumerate(prompts):
        logger.info(f"Evaluating prompt pair {idx + 1}/{len(prompts)}")
        for prompt_type in ["original", "attacked"]:
            score, gts, preds = eval(pair[f"{prompt_type}_prompt"], dataset, model, return_preds=True)  # type: ignore
            pair[f"{prompt_type}_score"] = score
            raw_data[prompt_type] = {"gts": gts, "preds": preds}

        log_results(pair)
        log_classification_reports(raw_data, dataset.get_inv_proj_dict())

    return prompts


def run_attack_eval_pipeline(config: dict):
    storage = StorageInterface(config["path"])
    combinations = get_combinations(config)

    for combination in combinations:
        if not config["overwrite"] and storage.exists(combination):
            logger.debug("Combination exists and overwrite disabled, skipping...")
            logger.debug(f"\nSkipped combination:\n{combination}\n")
            continue

        logger.info(f"\nCurrent combination:\n{combination}\n")

        dataset = Dataset(combination.dataset_name)
        dataset_name = dataset.get_original_name()

        model = create_augmentation(combination.model_name, combination.model_augmentation, dataset.get_proj_dict())

        if combination.task == "eval":
            prompts = Prompts.get_eval_prompts(dataset_name, storage, combination, config["eval_prompt_source"])
            result = run_evaluation(prompts, dataset, model)
        else:
            prompts = Prompts.get_attack_prompts(dataset_name, combination.prompt_type, config["attack_prompt_count"])
            result = run_attack(prompts, dataset, model, combination.attack_name)

        storage.save(combination, result)
        model.clear_memory()
