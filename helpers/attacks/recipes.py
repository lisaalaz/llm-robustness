# Adapted from the PromptBench library (https://github.com/microsoft/promptbench)

from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import LevenshteinEditDistance, MaxWordsPerturbed
from textattack.constraints.pre_transformation import InputColumnModification, RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import SBERT, UniversalSentenceEncoder
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapMaskedLM,
    WordSwapNeighboringCharacterSwap,
    WordSwapQWERTY,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
)

ATTACK_CONFIG = {
    "textfooler": {"max_candidates": 50, "min_word_cos_sim": 0.6, "min_sentence_cos_sim": 0.840845057},
    "textbugger": {"max_candidates": 5, "min_sentence_cos_sim": 0.8},
    "deepwordbug": {"levenshtein_edit_distance": 30},
    "bertattack": {"max_candidates": 48, "max_word_perturbed_percent": 1, "min_sentence_cos_sim": 0.8},
    "checklist": {"max_candidates": 5},
    "stresstest": {"max_candidates": 5},
    "charattack": {"min_sentence_cos_sim": 0.7},
    "wordattack": {"max_candidates": 15, "min_word_cos_sim": 0.6, "min_sentence_cos_sim": 0.7},
}


class AttackRecipes:
    @staticmethod
    def textbugger():  # character level
        transformation = CompositeTransformation(
            [
                WordSwapRandomCharacterInsertion(
                    random_one=True, letters_to_insert=" ", skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterDeletion(random_one=True, skip_first_char=True, skip_last_char=True),
                WordSwapNeighboringCharacterSwap(random_one=True, skip_first_char=True, skip_last_char=True),
                WordSwapHomoglyphSwap(),
                WordSwapEmbedding(max_candidates=ATTACK_CONFIG["textbugger"]["max_candidates"]),
            ]
        )

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(UniversalSentenceEncoder(threshold=ATTACK_CONFIG["textbugger"]["min_sentence_cos_sim"]))

        search_method = GreedyWordSwapWIR(wir_method="delete")

        return transformation, constraints, search_method

    @staticmethod
    def deepwordbug():  # character level
        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(),
                WordSwapRandomCharacterSubstitution(),
                WordSwapRandomCharacterDeletion(),
                WordSwapRandomCharacterInsertion(),
            ]
        )

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(LevenshteinEditDistance(ATTACK_CONFIG["deepwordbug"]["levenshtein_edit_distance"]))

        search_method = GreedyWordSwapWIR()
        return transformation, constraints, search_method

    @staticmethod
    def textfooler():  # word level
        transformation = WordSwapEmbedding(max_candidates=ATTACK_CONFIG["textfooler"]["max_candidates"])

        textfooler_stopwords = set(["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"])  # fmt: skip
        constraints = [RepeatModification(), StopwordModification(stopwords=textfooler_stopwords)]
        constraints.append(InputColumnModification(["premise", "hypothesis"], {"premise"}))
        constraints.append(WordEmbeddingDistance(min_cos_sim=ATTACK_CONFIG["textfooler"]["min_word_cos_sim"]))
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        constraints.append(
            UniversalSentenceEncoder(
                threshold=ATTACK_CONFIG["textfooler"]["min_sentence_cos_sim"],
                metric="angular",
                compare_against_original=False,
                window_size=15,
                skip_text_shorter_than_window=True,
            )
        )

        search_method = GreedyWordSwapWIR(wir_method="delete")
        return transformation, constraints, search_method

    @staticmethod
    def bertattack():  # word level
        transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=48)

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(MaxWordsPerturbed(max_percent=ATTACK_CONFIG["bertattack"]["max_word_perturbed_percent"]))
        constraints.append(
            UniversalSentenceEncoder(
                threshold=ATTACK_CONFIG["bertattack"]["min_sentence_cos_sim"],
                metric="cosine",
                compare_against_original=True,
                window_size=None,
            )
        )

        search_method = GreedyWordSwapWIR(wir_method="unk")
        return transformation, constraints, search_method

    @staticmethod
    def charattack():
        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(),
                WordSwapRandomCharacterSubstitution(),
                WordSwapRandomCharacterDeletion(),
                WordSwapRandomCharacterInsertion(),
                WordSwapQWERTY(),
            ]
        )

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(
            SBERT(
                threshold=ATTACK_CONFIG["charattack"]["min_sentence_cos_sim"],
                metric="cosine",
                model_name="all-mpnet-base-v2",
                compare_against_original=True,
                window_size=None,
            )
        )

        search_method = GreedyWordSwapWIR(wir_method="delete")
        return transformation, constraints, search_method

    @staticmethod
    def wordattack():
        transformation = WordSwapEmbedding(max_candidates=ATTACK_CONFIG["wordattack"]["max_candidates"])

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(WordEmbeddingDistance(min_cos_sim=ATTACK_CONFIG["wordattack"]["min_word_cos_sim"]))
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        constraints.append(
            SBERT(
                threshold=ATTACK_CONFIG["wordattack"]["min_sentence_cos_sim"],
                metric="cosine",
                model_name="all-mpnet-base-v2",
                compare_against_original=True,
                window_size=None,
            )
        )

        search_method = GreedyWordSwapWIR(wir_method="delete")
        return transformation, constraints, search_method
