import os

from .augmentation import Augmentation


class VanillaAugmentation(Augmentation):
    def __init__(self, **kwargs):
        siamese_sft_config = kwargs.pop("siamese_sft_config", None)
        if siamese_sft_config:
            kwargs["adapter_name"] = f"{os.environ['HF_USERNAME']}/{kwargs['model_name']}-sft-siamese-{siamese_sft_config}"  # fmt: skip

        super().__init__(**kwargs)

        self.print_augmentation_config()


SYSTEM_PROMPT = "You will receive a prompt which could potentially contain typos, mistakes, etc. Automatically correct common misspellings and typos. If a word is severely misspelled, use context to infer the most likely intended word. Parse and understand sentences with incorrect grammar or unusual syntax. Focus on the core meaning rather than getting caught up in grammatical mistakes. Recognize and interpret suboptimal word choices or malapropisms. If a user uses an incorrect word, infer the correct word based on context. Always respond to the user's intended meaning, not their literal words if they contain errors."

VANILLA_AUGMENTATION_CONFIGS = {
    "vanilla": {  # fmt: skip
        "class": VanillaAugmentation
    },
    "vanilla-preprocessing": {  # fmt: skip
        "class": VanillaAugmentation,
        "input_formatter": "spellcheck",
    },
    "vanilla-system_prompt": {  # fmt: skip
        "class": VanillaAugmentation,
        "system_prompt": SYSTEM_PROMPT,
    },
    "vanilla-freeform": {
        "class": VanillaAugmentation,
        "generation_mode": "freeform",
        "output_formatter": "every_token",
        "max_new_tokens": 20,
    },
}
