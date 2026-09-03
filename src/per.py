"""
Phoneme Error Rate (PER) implementation.

Pulled directly out of 02_phoneme_error_rate.ipynb, post-fix. The old
version of error_breakdown() had a duplicate block that silently
overwrote the hypothesis-side case normalisation; that bug is fixed
here and was re-verified against jiwer (6/6 match) before this file
was written.

Source of truth is the notebook. If you change behaviour here, re-run
02_phoneme_error_rate.ipynb top to bottom, confirm the jiwer
cross-check still shows 6/6 matches, before trusting downstream work.
"""

from g2p import tokenise_ipa


def edit_distance(seq1, seq2):
    """
    Compute the minimum number of edits (insert, delete, substitute)
    needed to turn seq1 into seq2. Wagner-Fischer algorithm.
    """
    m, n = len(seq1), len(seq2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )

    return dp[m][n]


def phoneme_error_rate(reference, hypothesis):
    """
    PER = edit_distance(reference, hypothesis) / len(reference)

    Handles:
    - ARPAbet lists: normalises to uppercase before comparing
    - IPA strings: tokenises properly using tokenise_ipa
    - Mixed input: converts strings to token lists automatically

    PER = 0.0  perfect pronunciation
    PER = 1.0  every phoneme wrong
    PER > 1.0  possible when hypothesis is much longer than reference
    """
    if isinstance(reference, str):
        ref = tokenise_ipa(reference)
    else:
        ref = [p.upper() for p in reference]

    if isinstance(hypothesis, str):
        hyp = tokenise_ipa(hypothesis)
    else:
        hyp = [p.upper() for p in hypothesis]

    if len(ref) == 0:
        return 0.0

    distance = edit_distance(ref, hyp)
    per = distance / len(ref)
    return round(per, 3)


def error_breakdown(reference, hypothesis):
    """
    Extended edit distance that tracks substitutions, deletions,
    and insertions separately.

    Uses the same Wagner-Fischer algorithm but records which
    operation was used at each step via backtracking.

    FIXED (was the Notebook 2 bug): hypothesis is normalised exactly
    once, matching how reference is normalised. There used to be a
    second, duplicate block here that overwrote hyp back to a raw,
    non-uppercased split, do not reintroduce it.
    """
    if isinstance(reference, str):
        ref = tokenise_ipa(reference)
    else:
        ref = [p.upper() for p in reference]

    if isinstance(hypothesis, str):
        hyp = tokenise_ipa(hypothesis)
    else:
        hyp = [p.upper() for p in hypothesis]

    m, n = len(ref), len(hyp)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion from ref
                    dp[i][j-1],    # insertion into hyp
                    dp[i-1][j-1]   # substitution
                )

    # Backtrack to count each error type
    substitutions = 0
    deletions = 0
    insertions = 0

    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    total = substitutions + deletions + insertions
    per = round(total / len(ref), 3) if len(ref) > 0 else 0.0

    return {
        "reference": ref,
        "hypothesis": hyp,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_errors": total,
        "per": per,
        "accuracy": round((1 - min(per, 1.0)) * 100)
    }


def print_breakdown(result):
    print(f"Reference:     {result['reference']}")
    print(f"Hypothesis:    {result['hypothesis']}")
    print(f"Substitutions: {result['substitutions']}")
    print(f"Deletions:     {result['deletions']}")
    print(f"Insertions:    {result['insertions']}")
    print(f"Total errors:  {result['total_errors']}")
    print(f"PER:           {result['per']}")
    print(f"Accuracy:      {result['accuracy']}/100")
    print("-" * 50)