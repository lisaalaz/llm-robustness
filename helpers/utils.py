import functools
import itertools
import logging
import os
import pprint
import random
import sys
import uuid
import warnings
from datetime import datetime

LOGGER_NAME = uuid.uuid4().hex
TRACE_LEVEL = 5

logger = logging.getLogger(LOGGER_NAME)


class InputFormatter:
    QUESTION_ANSWER_TEMPLATE = "\n{content}\nAnswer:"

    @staticmethod
    def spellcheck(prompt: str, spell_checker, excluded_words: set[str]) -> str:
        corrected_tokens = []
        tokens = tokenize(prompt)

        for token in tokens:
            if token.isalpha() and token.lower() not in excluded_words:
                corrected_tokens.append(spell_checker.correction(token))
            else:
                corrected_tokens.append(token)

        spellchecked_prompt = detokenize(corrected_tokens)
        logger.debug(f"Spellchecked prompt: {repr(spellchecked_prompt)}")
        return spellchecked_prompt

    @staticmethod
    def format_input(prompt: str, sample: str) -> str:
        prompt += InputFormatter.QUESTION_ANSWER_TEMPLATE
        return prompt.replace("{content}", sample)


class OutputFormatter:
    def __init__(self, output_formatter: str, proj_dict: dict[str, int], class_labels: list[str]):
        if output_formatter == "first_token":
            self.format = self.first_token
        elif output_formatter == "promptbench":
            self.format = self.promptbench
        elif output_formatter == "every_token":
            self.format = self.every_token
        else:
            raise ValueError(f"Output formatter {output_formatter} not found")

        self.proj_dict = proj_dict
        self.class_labels = class_labels

    @staticmethod
    def base_process(pred: str) -> str:
        return (
            pred.lower()
            .replace("<pad>", "")
            .replace("</s>", "")
            .replace("\n", " ")
            .strip(",._\"'-+=!?()&^*%$#@:\\|{}[]<>/`\t\r\v\f ")
        )

    def first_token(self, pred: str) -> int:
        processed = self.base_process(pred)
        for key in self.proj_dict:
            if processed.startswith(key):
                return self.proj_dict[key]
        logger.warning(f'Prediction: "{repr(pred)}" not found in {self.class_labels}')
        return -1

    def promptbench(self, pred: str) -> int:
        processed = self.base_process(pred).split(" ")[-1]
        if processed in self.proj_dict:
            return self.proj_dict[processed]
        logger.warning(f'Prediction: "{repr(pred)}" not found in {self.class_labels}')
        return -1

    def every_token(self, pred: str) -> int:
        for word in self.base_process(pred).split(" "):
            processed = self.base_process(word)
            if processed in self.proj_dict:
                return self.proj_dict[processed]
        logger.warning(f'Prediction: "{repr(pred)}" not found in {self.class_labels}')
        return -1


class Color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"

    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    END = "\033[0m"

    @staticmethod
    def format(string: str, *types: str) -> str:
        for type in types:
            if not type.startswith("\033["):
                type = getattr(Color, type.upper())
            string = f"{type}{string}{Color.END}"
        return string


def setup_logger(log_level: str) -> None:
    formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(LOGGER_NAME)
    logger.propagate = False
    logger.handlers.clear()
    logger.addHandler(handler)

    if log_level.upper() == "TRACE":
        logger.setLevel(TRACE_LEVEL)
        logging.getLogger().setLevel(TRACE_LEVEL)
    else:
        logger.setLevel(log_level)

    logger.info(f"Logger set to {log_level}")


def set_library_log_level() -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import datasets
    import evaluate
    import tensorflow as tf
    import transformers

    tf.get_logger().setLevel("ERROR")

    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()

    evaluate.logging.set_verbosity_error()
    evaluate.logging.disable_progress_bar()

    datasets.logging.set_verbosity_error()
    datasets.logging.disable_progress_bar()


def log_system_info() -> None:
    import GPUtil
    import psutil

    with open("/proc/cpuinfo", "r") as f:
        for line in f:
            if line.strip().startswith("model name"):
                cpu_name = line.split(":")[1].strip()
                logger.info(f"CPU: {cpu_name}")
                break

    gpus = GPUtil.getGPUs()
    if len(gpus) == 0:
        logger.warning("No GPUs found")
    else:
        logger.info(f"GPU: {gpus[0].name} x{len(gpus)}")

    virtual_mem = psutil.virtual_memory()
    logger.info(f"Memory: {virtual_mem.total / (1024 ** 3):.2f} GB")


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    logger.info(f"Seed set to {seed}")


def load_env() -> None:
    import dotenv

    dotenv.load_dotenv()
    logger.info("Loaded .env")


def setup(log_level: str, seed: int) -> None:
    warnings.simplefilter(action="ignore", category=FutureWarning)
    sys.setrecursionlimit(100000)

    setup_logger(log_level)
    load_env()
    set_library_log_level()
    log_system_info()
    seed_everything(seed)
    log_vram_usage()


def load_config(path: str) -> dict:
    import yaml

    if not os.path.exists(path) or not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found at {path}")

    if not path.endswith(".yaml") and not path.endswith(".yml"):
        raise ValueError("Config file must be a YAML file")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {path}")
    return config


def calculate_pdr(original_score: float, attacked_score: float) -> float:
    if original_score == 0:
        return 0
    return 1 - (attacked_score / original_score)


def log_results(data: dict[str, str | float]) -> None:
    logger.info(f"Original prompt: {repr(data['original_prompt'])}")
    logger.info(f"Adversarial prompt: {repr(data['attacked_prompt'])}")
    logger.info(f"Original score: {data['original_score']}")
    logger.info(f"Attacked score: {data['attacked_score']}")
    logger.info(f"PDR: {calculate_pdr(float(data['original_score']), float(data['attacked_score'])):.4f}\n")


def log_classification_reports(raw_data: dict[str, dict[str, list[int]]], inv_proj_dict: dict[int, str]):
    from sklearn.metrics import classification_report

    inv_proj_dict[-1] = "unknown"
    for prompt_type in ["original", "attacked"]:
        gts = [inv_proj_dict[x] for x in raw_data[prompt_type]["gts"]]
        preds = [inv_proj_dict[x] for x in raw_data[prompt_type]["preds"]]

        report = classification_report(gts, preds, zero_division=0)
        logger.info(f"{prompt_type.capitalize()} classification report:\n{report}")


def get_combinations(config: dict):
    from .storages import Combination

    return [
        Combination(config["task"], *c)
        for c in itertools.product(
            config["model_names"],
            config["model_augmentations"],
            config["dataset_names"],
            config["attack_names"],
            config["prompt_types"],
        )
    ]


def get_combinations_count(config: dict) -> int:
    return (
        len(config["model_names"])
        * len(config["model_augmentations"])
        * len(config["dataset_names"])
        * len(config["attack_names"])
        * len(config["prompt_types"])
    )


def named_product(**items) -> list[dict]:
    return [dict(zip(items.keys(), c)) for c in itertools.product(*items.values())]


def flatten(outer_list: list[list]) -> list:
    return [item for inner_list in outer_list for item in inner_list]


def tokenize(prompt: str) -> list[str]:
    import nltk

    tokens = nltk.word_tokenize(prompt)
    tokens = nltk.tokenize.MWETokenizer(
        [
            ("[", "UNK", "]"),
            ("[", "CLS", "]"),
            ("[", "SEP", "]"),
            ("<", "unk", ">"),
            ("<", "s", ">"),
            ("<", "/s", ">"),
            ("<", "pad", ">"),
            ("<", "mask", ">"),
        ],
        separator="",
    ).tokenize(tokens)

    logger.debug(f"Tokens: {tokens}")
    return tokens


def detokenize(tokens: list[str]) -> str:
    return "".join([" " + token if token not in ".,!?;:'" else token for token in tokens]).strip()


def time_function(message: str):
    def outer_wrapper(function):
        @functools.wraps(function)
        def inner_wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = function(*args, **kwargs)
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"{message}: {duration.seconds//3600}h {(duration.seconds//60)%60}m {duration.seconds%60}s")
            return result

        return inner_wrapper

    return outer_wrapper


def perplexity(perplexity_model):
    import uuid

    import evaluate

    metric = evaluate.load("perplexity", module_type="metric", experiment_id=uuid.uuid4().hex)
    return lambda predictions: metric.compute(predictions=predictions, model_id=perplexity_model)["perplexities"]  # type: ignore


def log_augmented_prompt(augmented_prompt: str | list[str], prompt: str):
    if isinstance(augmented_prompt, list):
        logger.debug(f"Augmented prompt:\n{pprint.pformat(augmented_prompt)}")
    elif augmented_prompt != prompt:
        logger.debug(f"Augmented prompt: {repr(augmented_prompt)}")
    else:
        logger.debug("Augmented prompt is the same as the original prompt")


def get_full_path(path: str) -> str:
    if "~" in path:
        return os.path.expanduser(path)
    return os.path.abspath(path)


def log_vram_usage():
    import torch

    mem_info = torch.cuda.mem_get_info()
    total_mem_mb = mem_info[1] / (1024**2)
    free_mem_mb = mem_info[0] / (1024**2)
    used_mem_mb = total_mem_mb - free_mem_mb
    used_mem_percent = (used_mem_mb / total_mem_mb) * 100

    logger.info(f"PyTorch GPU memory used: {used_mem_mb:.0f}/{total_mem_mb:.0f} MB ({used_mem_percent:.2f}%)")


def log_progress(iterable, desc, n_percent=10):
    from tqdm import tqdm

    progress_bar = tqdm(total=len(iterable), desc=desc, file=open(os.devnull, "w"))
    previous_percentage = 0

    for i, item in enumerate(iterable):
        yield item, progress_bar

        progress_bar.update(1)
        current_percentage = (i + 1) / len(iterable) * 100

        if int(current_percentage) // n_percent > previous_percentage // n_percent:
            previous_percentage = int(current_percentage)
            logger.info(str(progress_bar))

    progress_bar.close()
