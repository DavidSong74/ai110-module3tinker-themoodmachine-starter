# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
import string
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Maps emoji strings to placeholder tokens that score_text can treat as signals.
# Text-style emojis must be checked before punctuation is removed.
EMOJI_MAP: Dict[str, str] = {
    # text-style
    ":)":  "emoji_positive",
    ":-)": "emoji_positive",
    ":D":  "emoji_positive",
    ":-D": "emoji_positive",
    ":(":  "emoji_negative",
    ":-(": "emoji_negative",
    # unicode — labelled to match how people actually use them
    "😊": "emoji_positive",
    "😄": "emoji_positive",
    "😂": "emoji_positive",   # laughter / joy
    "😭": "emoji_positive",   # often used for overwhelming joy ("i'm shaking 😭")
    "❤️": "emoji_positive",
    "🥰": "emoji_positive",
    "🙃": "emoji_negative",   # sarcasm / passive frustration
    "😞": "emoji_negative",
    "😔": "emoji_negative",
    "😤": "emoji_negative",
    "💀": "emoji_mixed",      # humor / hyperbole ("i'm dead 💀")
    "🥲": "emoji_mixed",      # bittersweet
    "😮‍💨": "emoji_mixed",   # relief mixed with exhaustion
}

# Expands contractions so negation survives punctuation removal.
# e.g. "don't" -> "do not"  (the apostrophe would otherwise be stripped first)
CONTRACTIONS: Dict[str, str] = {
    "can't":   "cannot",
    "won't":   "will not",
    "don't":   "do not",
    "doesn't": "does not",
    "didn't":  "did not",
    "isn't":   "is not",
    "aren't":  "are not",
    "wasn't":  "was not",
    "weren't": "were not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "i'm":     "i am",
    "i've":    "i have",
    "i'd":     "i would",
    "i'll":    "i will",
    "it's":    "it is",
    "that's":  "that is",
    "there's": "there is",
    "they're": "they are",
    "we're":   "we are",
    "you're":  "you are",
}


# Words that flip the sentiment of the next token.
NEGATION_WORDS = {"not", "never", "no", "cannot"}

# Words that double the weight of the next sentiment token.
AMPLIFIER_WORDS = {"really", "so", "absolutely", "totally", "very", "extremely", "super"}

# Cap how many times any single word can contribute to the score.
# Prevents "happy happy happy" from dominating while still rewarding repetition.
MAX_WORD_FREQ = 2


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps:
          1. Detect ALL CAPS words (duplicated at the end for extra weight).
          2. Replace emoji strings with placeholder tokens (e.g. "emoji_positive").
          3. Lowercase.
          4. Expand contractions ("don't" -> "do not") so negation survives punctuation removal.
          5. Normalize repeated characters ("soooo" -> "soo").
          6. Remove punctuation.
          7. Split into tokens.
          8. Duplicate originally-ALL-CAPS tokens for double weight in score_text.
        """
        # Step 1: record which words are ALL CAPS before lowercasing.
        # These will be duplicated later so score_text counts them twice.
        caps_words = {
            w.strip(string.punctuation).lower()
            for w in text.split()
            if w.strip(string.punctuation).isupper()
            and len(w.strip(string.punctuation)) > 1
        }

        # Step 2: replace emoji strings with placeholder tokens before
        # punctuation removal destroys text-style emojis like ":)".
        for emoji, placeholder in EMOJI_MAP.items():
            text = text.replace(emoji, f" {placeholder} ")

        # Step 3: lowercase.
        text = text.lower()

        # Step 4: expand contractions before punctuation removal strips
        # apostrophes ("don't" -> "do not", so negation survives).
        for contraction, expansion in CONTRACTIONS.items():
            text = re.sub(rf"\b{re.escape(contraction)}\b", expansion, text)

        # Step 5: normalize repeated characters so emphasis is preserved
        # but words stay matchable ("soooo" -> "soo", "nooo" -> "noo").
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # Step 6: remove punctuation.
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Step 7: split into tokens.
        tokens = text.split()

        # Step 8: duplicate originally-ALL-CAPS tokens so score_text
        # gives them double weight with no changes required there.
        result = []
        for token in tokens:
            result.append(token)
            if token in caps_words:
                result.append(token)

        return result

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def _score_breakdown(self, text: str) -> Tuple[int, int]:
        """
        Core scoring pass. Returns (pos_score, neg_score) separately so
        predict_label can distinguish "mixed" (both > 0) from "neutral" (both == 0).

        Features applied per token:
          - Emoji placeholder tokens (from preprocess) → direct signal
          - Negation words (not, never, no, cannot) → flip next sentiment token
          - Amplifier words (really, very, absolutely…) → double next sentiment token
          - Frequency cap (MAX_WORD_FREQ) → each unique word contributes at most twice
          - ALL CAPS amplification is handled upstream in preprocess (token duplicated)
        """
        tokens = self.preprocess(text)
        pos_score = 0
        neg_score = 0
        word_counts: Dict[str, int] = {}

        pending_negate = False
        pending_amplify = 1

        for token in tokens:
            # --- negation trigger ---
            if token in NEGATION_WORDS:
                pending_negate = True
                continue

            # --- amplifier trigger ---
            if token in AMPLIFIER_WORDS:
                pending_amplify = 2
                continue

            # --- emoji placeholder tokens ---
            if token == "emoji_positive":
                pos_score += pending_amplify
                pending_negate = False
                pending_amplify = 1
                continue
            if token == "emoji_negative":
                neg_score += pending_amplify
                pending_negate = False
                pending_amplify = 1
                continue
            if token == "emoji_mixed":
                # no score change, but resets pending state
                pending_negate = False
                pending_amplify = 1
                continue

            # --- sentiment words ---
            is_positive = token in self.positive_words
            is_negative = token in self.negative_words

            if is_positive or is_negative:
                word_counts[token] = word_counts.get(token, 0) + 1
                if word_counts[token] <= MAX_WORD_FREQ:
                    signal = pending_amplify
                    if pending_negate:
                        # flip: "not happy" → negative, "not bad" → positive
                        if is_positive:
                            neg_score += signal
                        else:
                            pos_score += signal
                    else:
                        if is_positive:
                            pos_score += signal
                        else:
                            neg_score += signal
                pending_negate = False
                pending_amplify = 1
            else:
                # unrecognised word resets pending state so negation/amplification
                # doesn't accidentally carry across unrelated words
                pending_negate = False
                pending_amplify = 1

        return pos_score, neg_score

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.
        Delegates to _score_breakdown and returns pos_score - neg_score.
        A positive result means more positive signal; negative means more negative.
        """
        pos_score, neg_score = self._score_breakdown(text)
        return pos_score - neg_score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        pos_score, neg_score = self._score_breakdown(text)

        if pos_score > 0 and neg_score > 0:
            return "mixed"
        if pos_score > neg_score:
            return "positive"
        if neg_score > pos_score:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
