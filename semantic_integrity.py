import pprint

import torch
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig

from helpers.utils import get_combinations, load_config, load_env

load_env()

from helpers.storages import StorageInterface

config = load_config("config.yaml")
print(f"Config:\n{pprint.pformat(config, sort_dicts=False)}")

storage = StorageInterface(config["path"])
combinations = get_combinations(config)
model = SentenceTransformer(
    "intfloat/e5-mistral-7b-instruct",
    cache_folder="/vol/bitbucket/aa6923/st",
    model_kwargs={
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    },
)

for combination in combinations:
    pairs = storage.load(combination)

    if pairs is None:
        continue

    similarity = 0
    for pair in pairs:
        embeddings = model.encode(
            [pair["original_prompt"], pair["attacked_prompt"]],
            normalize_embeddings=True,
            convert_to_tensor=True,
            prompt_name="sts_query",
        )
        pair["cosine_similarity"] = model.similarity_pairwise(
            embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
        ).item()
        similarity += pair["cosine_similarity"]

    print(f"Combination: {combination}, average cosine similarity: {similarity/len(pairs)}")
    storage.save(combination, pairs)
