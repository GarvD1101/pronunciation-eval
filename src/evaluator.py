"""
End-to-end pronunciation evaluation pipeline.

Pulled directly out of 01_g2p_and_ipa.ipynb, post-fix. get_reference()
used to build the phonemizer fallback with raw list(ipa), which split
diacritics off as spurious standalone tokens. Fixed to use
tokenise_ipa(ipa) instead, verified against Notebook 2's jiwer
cross-check (6/6 match) before this file was written.

Source of truth is the notebook. If you change behaviour here, re-run
01_g2p_and_ipa.ipynb top to bottom before trusting downstream work.
"""

import nltk
from nltk.corpus import cmudict
from phonemizer import phonemize

from g2p import tokenise_ipa, arpabet_to_ipa, strip_tie_bars
from per import phoneme_error_rate

nltk.download('cmudict', quiet=True)


class PronunciationEvaluator:
    """
    Given a word and a "spoken" phoneme sequence, tells you how
    accurately the word was pronounced.

    In a real system, the spoken phonemes come from running audio
    through a phoneme recogniser like allosaurus. Notebook 3 is
    where that gets wired in, spoken_phonemes stops being manual
    input and starts being model.recognize(audio_path).split().
    """

    def __init__(self):
        self.cmu = cmudict.dict()
        print("PronunciationEvaluator ready.")

    def get_reference(self, word):
        """
        Get the reference (correct) phonemes for a word, always as IPA.

        CMUdict is ARPAbet natively, converted here via arpabet_to_ipa()
        so it's directly comparable to phonemizer output and to real
        audio recognised by allosaurus, both of which are IPA. Without
        this conversion, any CMUdict-covered word scores badly against
        real audio regardless of how well it was actually pronounced,
        the reference and hypothesis would be in different alphabets.
        """
        result = self.cmu.get(word.lower().strip())
        if result:
            return {
                "phonemes": arpabet_to_ipa(result[0]),
                "source": "CMUdict",
                "notation": "IPA (converted from ARPAbet)",
                "confidence": "HIGH"
            }

        ipa = phonemize(
            word,
            backend="espeak",
            language="en-us",
            strip=True
        )
        return {
            "phonemes": tokenise_ipa(ipa),
            "source": "phonemizer",
            "notation": "IPA",
            "confidence": "LOW"
        }

    def evaluate(self, word, spoken_phonemes):
        """
        Evaluate how accurately a word was pronounced.

        word:            the text of the word
        spoken_phonemes: what was actually said, as a phoneme list
        """
        reference = self.get_reference(word)
        # Normalise both sides the same way, applying it to only one
        # side is exactly the bug that broke Notebook 2's error_breakdown()
        ref_phonemes = strip_tie_bars(reference["phonemes"])
        spoken_phonemes = strip_tie_bars(spoken_phonemes)

        per = phoneme_error_rate(ref_phonemes, spoken_phonemes)

        score = round((1 - min(per, 1.0)) * 100)

        if per == 0.0:
            grade = "PERFECT"
        elif per <= 0.2:
            grade = "GOOD"
        elif per <= 0.5:
            grade = "ACCEPTABLE"
        else:
            grade = "POOR"

        return {
            "word": word,
            "reference_phonemes": ref_phonemes,
            "spoken_phonemes": spoken_phonemes,
            "reference_source": reference["source"],
            "reference_confidence": reference["confidence"],
            "per_score": per,
            "accuracy_score": score,
            "grade": grade
        }

    def print_report(self, result):
        """Pretty print an evaluation result."""
        print(f"\nWord:           {result['word']}")
        print(f"Reference:      {result['reference_phonemes']}")
        print(f"  Source:       {result['reference_source']} "
              f"(confidence: {result['reference_confidence']})")
        print(f"Spoken:         {result['spoken_phonemes']}")
        print(f"PER:            {result['per_score']}")
        print(f"Accuracy:       {result['accuracy_score']}/100")
        print(f"Grade:          {result['grade']}")
        print("-" * 50)