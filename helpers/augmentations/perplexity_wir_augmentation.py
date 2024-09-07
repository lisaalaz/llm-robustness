import logging
import pprint

import nltk
import numpy as np

from helpers.utils import LOGGER_NAME, detokenize, perplexity, tokenize

from .augmentation import Augmentation

logger = logging.getLogger(LOGGER_NAME)


class PerplexityWIRAugmentation(Augmentation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # kwargs
        self.search_method: str = kwargs.pop("search_method")
        self.perplexity_model: str = kwargs.pop("perplexity_model")
        self.iterations: int = kwargs.pop("iterations")
        self.top_k: int = kwargs.pop("top_k")

        if self.search_method == "greedy_search":
            self.early_stopping: bool = kwargs.pop("early_stopping")
        elif self.search_method == "beam_search":
            self.beam_width: int = kwargs.pop("beam_width")

        # init
        self.compute_perplexity = perplexity(self.perplexity_model)
        self.stopwords = set(nltk.corpus.stopwords.words("english"))

    def augmentation(self, prompt: str) -> str | list[str]:
        if self.search_method == "greedy_search":
            return self.greedy_search(prompt)
        elif self.search_method == "beam_search":
            return self.beam_search(prompt)
        else:
            raise ValueError(f"Invalid search method: {self.search_method}")

    def perplexity_wir_augmentation(self, words: list[str], word_idx: int) -> list[str]:
        raise NotImplementedError

    def greedy_search(self, prompt: str) -> str:
        seen_word_idxs = set()
        prompt_perplexity = self.compute_perplexity(predictions=[prompt])[0]
        logger.debug(f"Original prompt perplexity: {prompt_perplexity:.4f}\n")

        for i in range(self.iterations):
            logger.debug(f"Iteration {i+1}/{self.iterations}")

            words = tokenize(prompt)

            word_idx = self.word_importance_ranking(words, seen_word_idxs)
            if word_idx is None:
                return prompt

            seen_word_idxs.add(word_idx)
            candidates = self.get_candidates(words, word_idx)
            if candidates is None:
                break
            candidate_prompts, candidate_perplexities = candidates

            best_prompt = candidate_prompts[np.argmin(candidate_perplexities)]
            best_prompt_perplexity = min(candidate_perplexities)

            if self.early_stopping and prompt_perplexity <= best_prompt_perplexity:
                logger.debug("No improvement in perplexity, stopping early\n")
                break
            elif prompt_perplexity <= best_prompt_perplexity:
                logger.debug("No improvement in perplexity, continuing with prompt from previous iteration\n")
                continue
            else:
                logger.debug(f"Prompt: {repr(prompt)} -> {repr(best_prompt)}")
                logger.debug(f"Perplexity: {prompt_perplexity:.4f} -> {best_prompt_perplexity:.4f}\n")
                prompt = best_prompt
                prompt_perplexity = best_prompt_perplexity

        return prompt

    def beam_search(self, prompt: str) -> str | list[str]:
        seen_word_idxs = set()

        prompts = np.array([prompt])
        prompt_perplexities = np.array(self.compute_perplexity(predictions=prompts))
        logger.debug(f"Original prompt perplexity: {prompt_perplexities[0]:.4f}\n")

        for i in range(self.iterations):
            logger.debug(f"Iteration {i+1}/{self.iterations}")

            length = len(prompts)

            for j, prompt in enumerate(prompts.copy()):
                logger.debug(f"Prompt {j+1}/{length}")

                words = tokenize(prompt)

                word_idx = self.word_importance_ranking(words, seen_word_idxs)
                if word_idx is None:
                    return prompts[0]

                seen_word_idxs.add(word_idx)
                candidates = self.get_candidates(words, word_idx)
                if candidates is None:
                    break
                candidate_prompts, candidate_perplexities = candidates

                prompts = np.append(prompts, candidate_prompts)
                prompt_perplexities = np.append(prompt_perplexities, candidate_perplexities)

            best_idxs = np.argsort(prompt_perplexities)[: self.beam_width]
            prompts = prompts[best_idxs]
            prompt_perplexities = prompt_perplexities[best_idxs]

            print_str = pprint.pformat(list(zip(prompts, prompt_perplexities)), width=120)
            logger.debug(f"Best prompts and perplexities:\n{print_str}\n")

        return prompts[0]

    def get_candidates(self, words: list[str], word_idx: int):
        logger.debug(f"Most important word: {repr(words[word_idx])}")

        candidate_words = self.perplexity_wir_augmentation(words, word_idx)
        if not candidate_words:
            return None

        candidate_prompts = [detokenize(words[:word_idx] + [c] + words[word_idx + 1 :]) for c in candidate_words]
        candidate_perplexities = self.compute_perplexity(predictions=candidate_prompts)

        print_str = pprint.pformat(list(zip(candidate_prompts, candidate_perplexities)), width=120)
        logger.debug(f"Candidate prompts and perplexities:\n{print_str}")

        return candidate_prompts, candidate_perplexities

    def word_importance_ranking(self, words: list[str], seen_word_idxs: set[int]):
        prompts = []
        idxs = []

        for idx, word in enumerate(words):
            if (
                not word.isalpha()
                or word.lower() in self.excluded_words
                or word.lower() in self.stopwords
                or idx in seen_word_idxs
            ):
                continue
            prompts.append(detokenize(words[:idx] + words[idx + 1 :]))
            idxs.append(idx)

        if not prompts:
            return None

        scores = self.compute_perplexity(predictions=prompts)

        print_str = pprint.pformat(list(zip(prompts, scores)), width=120)
        logger.debug(f"WIR (word deletion) - prompts and perplexities:\n{print_str}")

        return idxs[np.argmax(scores)]
