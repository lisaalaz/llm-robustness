import logging
import os
import pprint
import tempfile

import torch
from datasets import Dataset as HuggingFaceDataset
from peft import LoraConfig, TaskType  # type: ignore
from transformers import TrainerCallback, TrainerControl, TrainerState
from trl import SFTConfig, SFTTrainer

from helpers.augmentations import get_sft_config
from helpers.models import Model, create_model
from helpers.storages import get_sft_dataset
from helpers.utils import LOGGER_NAME, log_progress

logger = logging.getLogger(LOGGER_NAME)


class LossThresholdCallback(TrainerCallback):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.log_history and "loss" in state.log_history[-1] and state.log_history[-1]["loss"] < self.threshold:
            control.should_training_stop = True


def get_target_modules(hidden_state_layer: int):
    if hidden_state_layer == -1:
        return ["q_proj", "v_proj"]

    target_modules = []

    for i in range(hidden_state_layer):
        target_modules.append(f"model.layers.{i}.self_attn.q_proj")
        target_modules.append(f"model.layers.{i}.self_attn.v_proj")

    return target_modules


def get_hidden_state(wrapper: Model, prompt: str, hidden_state_layer: int) -> torch.Tensor:
    original_input_ids = wrapper.tokenizer(prompt, return_tensors="pt").input_ids

    if wrapper.task_type is not TaskType.SEQ_2_SEQ_LM:
        outputs = wrapper.model(input_ids=original_input_ids, output_hidden_states=True)
        return torch.mean(outputs.hidden_states[hidden_state_layer], dim=1)

    outputs = wrapper.model(input_ids=original_input_ids, labels=original_input_ids, output_hidden_states=True)
    return torch.mean(outputs.encoder_hidden_states[-1], dim=1)


def run_siamese(wrapper: Model, hub_model_id: str, sft_config: dict, sft_dataset: HuggingFaceDataset):
    adapter_parameters = [parameter for parameter in wrapper.model.parameters() if parameter.requires_grad]

    optimizer = torch.optim.AdamW(adapter_parameters, lr=sft_config["learning_rate"])
    criterion = torch.nn.CosineSimilarity()
    epochs = sft_config["epochs"]
    loss_threshold = sft_config["early_stopping_loss_threshold"]
    previous_loss = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        for instance, pbar in log_progress(sft_dataset, desc=f"Epoch {epoch + 1}/{epochs}"):
            assert isinstance(instance, dict)

            with torch.no_grad(), wrapper.model.disable_adapter():
                original_hidden = get_hidden_state(wrapper, instance["original"], sft_config["hidden_state_layer"])

            optimizer.zero_grad()
            attacked_hidden = get_hidden_state(wrapper, instance["attacked"], sft_config["hidden_state_layer"])

            loss = 1 - criterion(original_hidden, attacked_hidden).mean()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        total_loss /= len(sft_dataset)
        loss_change = (100 * (total_loss - previous_loss) / previous_loss) if previous_loss != 0.0 else 0.0
        logger.info(f"Epoch {epoch + 1}/{epochs} - loss: {total_loss:.3f} ({loss_change:.2f}%)")
        previous_loss = total_loss

        if total_loss / len(sft_dataset) < loss_threshold:
            logger.info(f"Early stopping - loss = {total_loss:.3f}, threshold = {loss_threshold}")
            break

    wrapper.model.push_to_hub(hub_model_id)  # type: ignore


def run_rewriting(wrapper: Model, hub_model_id: str, sft_config: dict, sft_dataset: HuggingFaceDataset):
    assert wrapper.task_type == TaskType.CAUSAL_LM

    output_dir = tempfile.mkdtemp(dir=os.environ["SFT_DATA"])
    args = SFTConfig(
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        max_seq_length=1024,
        num_train_epochs=sft_config["epochs"],
        per_device_train_batch_size=sft_config["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_config["gradient_accumulation_steps"],
        learning_rate=sft_config["learning_rate"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )  # type: ignore

    trainer = SFTTrainer(
        model=wrapper.model,
        train_dataset=sft_dataset,
        args=args,
        callbacks=[LossThresholdCallback(threshold=sft_config["early_stopping_loss_threshold"])],
    )

    trainer.train()  # type: ignore
    trainer.push_to_hub()  # type: ignore


def run_sft_pipeline(config: dict, task: str):
    assert task in ["rewriting", "siamese"]

    sft_config_name = config[f"sft_{task}_config"]

    sft_config = sft_defaults(SFT_CONFIGS[task][sft_config_name])
    logger.info(f"SFT config:\n{pprint.pformat(sft_config)}\n")

    sft_dataset = get_sft_dataset(task, sft_config)
    logger.info(f"Dataset:\n{pprint.pformat(sft_dataset[:5])}\n")
    logger.info(f"Dataset size: {len(sft_dataset)}")

    for model_name in config["model_names"]:
        logger.info(f"Current model: {model_name}")
        model = create_model(model_name, model_context=f"{task.capitalize()} SFT", **get_sft_config())

        hub_model_id = f"{os.environ['HF_USERNAME']}/{model_name}-sft-{task}-{sft_config_name}"

        lora_config_kwargs = {
            "lora_alpha": sft_config["lora_alpha"],
            "lora_dropout": sft_config["lora_dropout"],
            "r": sft_config["lora_r"],
            "use_dora": sft_config["lora_use_dora"],
            "task_type": model.task_type,
        }

        if model.task_type is not TaskType.SEQ_2_SEQ_LM and task == "siamese":
            lora_config_kwargs["target_modules"] = get_target_modules(sft_config["hidden_state_layer"])

        peft_config = LoraConfig(**lora_config_kwargs)

        model.convert_to_peft(peft_config)

        if task == "rewriting":
            run_rewriting(model, hub_model_id, sft_config, sft_dataset)
        else:
            run_siamese(model, hub_model_id, sft_config, sft_dataset)

        model.clear_memory()


def sft_defaults(config):
    return {
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "lora_r": 64,
        "lora_use_dora": False,
        "epochs": 10,
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "early_stopping_loss_threshold": 0.0,
    } | config


SFT_CONFIGS = {
    "siamese": {
        # llama hidden_state_layer = 15
        "promptbench": {  # fmt: skip
            "dataset_version": "promptbench",
            "hidden_state_layer": 15,
        },
        "promptbench-attackaware": {  # fmt: skip
            "dataset_version": "promptbench-attackaware",
            "hidden_state_layer": 15,
        },
        "promptbench-attackblind": {  # fmt: skip
            "dataset_version": "promptbench-attackblind",
            "hidden_state_layer": 15,
        },
        "advglue": {  # fmt: skip
            "dataset_version": "advglue",
            "hidden_state_layer": 15,
        },
        "advglue-promptbench": {  # fmt: skip
            "dataset_version": "advglue-promptbench",
            "hidden_state_layer": 15,
        },
        # llama hidden_state_layer = -1
        "promptbench-old": {  # fmt: skip
            "dataset_version": "promptbench",
            "hidden_state_layer": -1,
        },
        "promptbench-attackaware-old": {  # fmt: skip
            "dataset_version": "promptbench-attackaware",
            "hidden_state_layer": -1,
        },
        "promptbench-attackblind-old": {  # fmt: skip
            "dataset_version": "promptbench-attackblind",
            "hidden_state_layer": -1,
        },
        "advglue-old": {  # fmt: skip
            "dataset_version": "advglue",
            "hidden_state_layer": -1,
        },
    },
    "rewriting": {
        # zeroshot
        "zs-promptbench-attackaware": {  # fmt: skip
            "rewriting_template": "zeroshot",
            "dataset_version": "promptbench-attackaware",
        },
        "zs-promptbench-attackblind": {  # fmt: skip
            "rewriting_template": "zeroshot",
            "dataset_version": "promptbench-attackblind",
        },
        # fewshot
        "fs-promptbench": {  # fmt: skip
            "rewriting_template": "fewshot",
            "dataset_version": "promptbench",
        },
        "fs-promptbench-attackaware": {  # fmt: skip
            "rewriting_template": "fewshot",
            "dataset_version": "promptbench-attackaware",
        },
        "fs-promptbench-attackblind": {  # fmt: skip
            "rewriting_template": "fewshot",
            "dataset_version": "promptbench-attackblind",
        },
        "fs-advglue": {  # fmt: skip
            "rewriting_template": "fewshot",
            "dataset_version": "advglue",
        },
        "fs-advglue-promptbench": {  # fmt: skip
            "rewriting_template": "fewshot",
            "dataset_version": "advglue-promptbench",
        },
    },
}
