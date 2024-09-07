import logging
import pprint
import warnings

from peft import TaskType  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from helpers.utils import LOGGER_NAME, TRACE_LEVEL

from .model import Model

logger = logging.getLogger(LOGGER_NAME)


class LlamaModel(Model):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=self.torch_dtype, device_map=self.device_map
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
        )
        self.task_type = TaskType.CAUSAL_LM

        self.set_logits_processor()
        self.print_model_config()

    def predict(self, input_text: str, **kwargs) -> str:
        if kwargs:
            logger.log(TRACE_LEVEL, f"Additional kwargs:\n{pprint.pformat(kwargs)}")

        temperature = kwargs.get("temperature", self.temperature)
        generation_mode = kwargs.get("generation_mode", self.generation_mode)
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        system_prompt = kwargs.get("system_prompt", self.system_prompt)

        logger.log(TRACE_LEVEL, f"Input: {repr(input_text)}")
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": input_text}]
        else:
            messages = [{"role": "user", "content": input_text}]

        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)  # type: ignore
        inputs = self.tokenizer(input_text, return_tensors="pt").input_ids

        generate_kwargs = {
            "inputs": inputs,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True if temperature > 0 else False,
        }

        if generation_mode == "constrained":
            generate_kwargs["logits_processor"] = self.logits_processor_list
            generate_kwargs["repetition_penalty"] = self.repetition_penalty

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            outputs = self.model.generate(**generate_kwargs)

        if generation_mode == "constrained":
            self.logits_processor_list[0].reset()  # type: ignore

        output_text = self.tokenizer.decode(outputs[:, inputs.shape[1] :][0], skip_special_tokens=True)
        logger.log(TRACE_LEVEL, f"Output: {repr(output_text)}")
        return output_text
