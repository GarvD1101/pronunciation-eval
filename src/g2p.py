"""
Grapheme-to-phoneme (G2P) utilities.

Pulled directly out of 01_g2p_and_ipa.ipynb after the tokenise_ipa fix
was verified (2 hops: CMUdict first, phonemizer/espeak fallback).

Source of truth is the notebook. If you change behaviour here, re-run
01_g2p_and_ipa.ipynb top to bottom and confirm the printed outputs
still match before trusting downstream notebooks.
"""

import nltk
from nltk.corpus import cmudict
from phonemizer import phonemize

nltk.download('cmudict', quiet=True)
cmu = cmudict.dict()


def tokenise_ipa(ipa_string):
    """
    Groups IPA diacritics with their base phoneme.
    Prevents diacritics from being counted as standalone sounds.
    Example: 'saiSHax' (length mark attached) -> one token per real phoneme,
    not one token per unicode character.
    """
    DIACRITICS = set(['\u02d0', '\u02b0', '\u02b7', '\u02b2', '\u0303', '\u0308', '\u02d1'])
    tokens = []
    for char in ipa_string:
        if char in DIACRITICS and tokens:
            tokens[-1] = tokens[-1] + char
        else:
            tokens.append(char)
    return tokens


def lookup_cmu(word):
    """
    Look up a word's phonemes in CMUdict.

    Returns the first (most common) pronunciation if found.
    Returns None if the word is not in the dictionary.
    """
    word = word.lower().strip()
    result = cmu.get(word)

    if result:
        return result[0]   # First pronunciation only
    else:
        return None        # Word not found


def phonemize_word(word):
    """
    Use phonemizer with espeak-ng backend to convert
    a word to IPA (International Phonetic Alphabet).

    backend="espeak" uses espeak-ng under the hood.
    language="en-us" tells it to use American English rules.
    strip=True removes extra whitespace from output.
    """
    result = phonemize(
        word,
        backend="espeak",
        language="en-us",
        strip=True
    )
    return result


def g2p(word):
    """
    Combined G2P function.

    Strategy:
    1. Try CMUdict first. It's fast, zero compute, and accurate
       for the 123k words it knows.
    2. If CMUdict misses, fall back to phonemizer/espeak.
       It handles unknown words including names, but with
       lower accuracy for non-English origins.

    Returns a dict with the phonemes, which source was used,
    and the notation type so we know what we're looking at.
    """
    word_clean = word.lower().strip()

    # Try CMUdict first
    cmu_result = cmu.get(word_clean)
    if cmu_result:
        return {
            "word": word,
            "phonemes": cmu_result[0],
            "source": "CMUdict",
            "notation": "ARPAbet",
            "found": True
        }

    # Fall back to phonemizer
    ipa_result = phonemize(
        word,
        backend="espeak",
        language="en-us",
        strip=True
    )
    return {
        "word": word,
        "phonemes": tokenise_ipa(ipa_result),   # properly tokenised
        "source": "phonemizer/espeak",
        "notation": "IPA",
        "found": False
    }


# ---------------------------------------------------------------------
# ARPAbet -> IPA
#
# CMUdict is ARPAbet, phonemizer and allosaurus are both IPA. Comparing
# an ARPAbet reference against an IPA hypothesis never matches, even
# for a perfect pronunciation, they're different alphabets, not just
# different notations of the same one. This converts CMUdict output
# into IPA so it can be fairly compared against real audio.
# ---------------------------------------------------------------------

ARPABET_TO_IPA = {
    # consonants
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'F': 'f', 'G': 'ɡ',
    'HH': 'h', 'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n',
    'NG': 'ŋ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ', 'T': 't',
    'TH': 'θ', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
    # vowels (default realisation; AH has a stress-conditioned override below)
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ',
    'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'IH': 'ɪ', 'IY': 'i', 'OW': 'oʊ',
    'OY': 'ɔɪ', 'UH': 'ʊ', 'UW': 'u',
}


def arpabet_to_ipa(arpabet_phonemes):
    """
    Convert a list of ARPAbet phonemes (CMUdict style, e.g.
    ['HH', 'AH0', 'L', 'OW1']) into IPA.

    Two ARPAbet quirks handled here:
    - Stress digits (0/1/2) are stripped before lookup. ARPAbet marks
      stress on the vowel itself, IPA in this project doesn't track it.
    - AH0 specifically maps to schwa 'ə', not 'ʌ'. Unstressed "uh" and
      stressed "uh" are genuinely different vowels in English, compare
      the second syllable of "sofa" to the vowel in "cup". CMUdict
      lumps both under AH and only the stress digit distinguishes them,
      so this needs a special case rather than a flat lookup.
    """
    ipa = []
    for phoneme in arpabet_phonemes:
        base = phoneme.rstrip('012')
        stress = phoneme[len(base):]

        if base == 'AH' and stress == '0':
            ipa.append('ə')
        else:
            ipa.append(ARPABET_TO_IPA.get(base, base))
    return ipa


def strip_tie_bars(ipa_tokens):
    """
    Splits affricates that allosaurus joins with a combining tie bar
    (U+0361), e.g. 'd\u0361ʒ' -> 'd', 'ʒ' as two separate tokens.

    Without this, allosaurus's affricate notation (one fused symbol)
    never matches phonemizer/CMUdict's convention (two plain letters)
    even when the actual sound is identical, the same category of
    problem as the case-sensitivity bug in Notebook 2: two systems
    disagreeing on notation, not on the actual sound.
    """
    expanded = []
    for token in ipa_tokens:
        if '\u0361' in token:
            expanded.extend(token.split('\u0361'))
        else:
            expanded.append(token)
    return expanded