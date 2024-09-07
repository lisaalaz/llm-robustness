import gc
import logging

import gensim.downloader
from gensim.models import KeyedVectors

from helpers.augmentations.perplexity_wir_augmentation import PerplexityWIRAugmentation
from helpers.utils import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class EmbeddingSynonymAugmentation(PerplexityWIRAugmentation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # kwargs
        self.embedding_model: str = kwargs.pop("embedding_model")

        # init
        self.embeddings: KeyedVectors = gensim.downloader.load(self.embedding_model)  # type: ignore

        self.print_augmentation_config()

    def perplexity_wir_augmentation(self, words: list[str], word_idx: int) -> list[str]:
        if words[word_idx] not in self.embeddings:
            logger.debug(f"Word {repr(words[word_idx])} not in vocabulary")
            return []
        raw_synonyms = self.embeddings.most_similar(words[word_idx], topn=self.top_k)  # type: ignore
        synonyms = [x[0] for x in raw_synonyms]
        logger.debug(f"Synonyms for {repr(words[word_idx])}: {synonyms}")
        return synonyms

    def clear_memory(self):
        del self.embeddings
        gc.collect()
        return super().clear_memory()


EMBEDDING_SYNONYM_AUGMENTATION_CONFIGS = {
    "embedding_synonym-beam_search-word2vec": {
        "class": EmbeddingSynonymAugmentation,
        "search_method": "beam_search",
        "beam_width": 5,
        "perplexity_model": "gpt2",
        "iterations": 10,
        "top_k": 5,
        "embedding_model": "word2vec-google-news-300",
    },
    "embedding_synonym-greedy_search-word2vec": {
        "class": EmbeddingSynonymAugmentation,
        "search_method": "greedy_search",
        "early_stopping": True,
        "perplexity_model": "gpt2",
        "iterations": 10,
        "top_k": 5,
        "embedding_model": "word2vec-google-news-300",
    },
}
