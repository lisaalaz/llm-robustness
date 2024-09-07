from helpers.utils import time_function

from .attack_eval_pipeline import run_attack_eval_pipeline
from .sft_pipeline import run_sft_pipeline


@time_function(message="Total duration")
def run_pipeline(config: dict) -> None:
    if config["task"].startswith("sft-"):
        run_sft_pipeline(config, config["task"].split("-")[-1])
    elif config["task"] in ["eval", "attack"]:
        run_attack_eval_pipeline(config)
    else:
        raise ValueError(f"Task {config['task']} not found")
