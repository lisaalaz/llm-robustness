import logging

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from helpers.augmentations.perplexity_wir_augmentation import PerplexityWIRAugmentation
from helpers.models import create_model
from helpers.utils import LOGGER_NAME, detokenize

logger = logging.getLogger(LOGGER_NAME)


class ContextualSynonymAugmentation(PerplexityWIRAugmentation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # kwargs
        self.embedding_model_name: str = kwargs.pop("embedding_model")

        # init
        self.model_kwargs = {
            "max_new_tokens": 10,
            "temperature": 0.0,
            "generation_mode": "freeform",
            "system_prompt": None,
        }
        if self.embedding_model_name == self.model_name:
            self.embedding_model = self.model
        else:
            self.embedding_model = create_model(
                self.embedding_model_name, model_context="embedding", **self.model_kwargs
            )
        self.filtered_vocab = self.filter_vocabulary()
        self.print_augmentation_config()

    def perplexity_wir_augmentation(self, words: list[str], word_idx: int) -> list[str]:
        word_emb = self.get_word_embedding(words, word_idx)

        vocab_embs = []
        vocab_words = []

        for vocab_word in self.filtered_vocab:
            if vocab_word.isalpha() and vocab_word not in self.excluded_words and vocab_word not in self.stopwords:
                temp_words = words.copy()
                temp_words[word_idx] = vocab_word
                vocab_embs.append(self.get_word_embedding(temp_words, word_idx))
                vocab_words.append(vocab_word)

        similarities = cosine_similarity([word_emb], vocab_embs)[0]  # type: ignore
        top_idxs = np.argsort(similarities)[-self.top_k - 1 : -1]
        synonyms = [vocab_words[i] for i in top_idxs]

        logger.debug(f"Synonyms for {repr(words[word_idx])}: {synonyms}")

        return synonyms

    def get_word_embedding(self, words: list[str], word_idx: int):
        inputs = self.embedding_model.tokenizer(detokenize(words), return_tensors="pt").input_ids

        with torch.no_grad():
            outputs = self.embedding_model.model(inputs, output_hidden_states=True)

        word_tokens = self.embedding_model.tokenizer.encode(words[word_idx], add_special_tokens=False)
        word_emb = outputs.encoder_last_hidden_state[word_idx + 1 : word_idx + 1 + len(word_tokens)].mean(dim=0).numpy()
        return word_emb

    def filter_vocabulary(self):
        vocab = []
        for vocab_word in self.embedding_model.tokenizer.vocab.keys():  # type: ignore
            word = vocab_word.removeprefix("Ġ")
            if not word.isalpha() or word in self.excluded_words or word in self.stopwords:
                continue
            vocab.append(word)
        return vocab

    def clear_memory(self):
        self.embedding_model.clear_memory()
        return super().clear_memory()


CONTEXTUAL_SYNONYM_AUGMENTATION_CONFIGS = {
    "contextual_synonym-beam_search-distilroberta": {
        "class": ContextualSynonymAugmentation,
        "search_method": "beam_search",
        "beam_width": 5,
        "perplexity_model": "gpt2",
        "iterations": 10,
        "top_k": 5,
        "embedding_model": "distilroberta-base",
    }
}
