"""Seeded, deterministic text perturbations for robustness evaluation.

Every perturbation is a pure function of ``(text, kind, seed)``: the same
arguments always produce the same string, in this process and in any other.
Randomness is drawn from a ``random.Random`` seeded with a SHA-256 digest of
those three values, so it does not depend on Python's salted ``hash()`` or on
call order (perturbing row 7 first or last gives the same result).

Kinds
-----
``typo``       one character-level typo (deletion, duplication or letter swap)
``char_swap``  two adjacent characters transposed inside a word
``abbrev``     words contracted to chat shorthand  (please -> pls, order -> ord)
``abbrev_expand`` shorthand expanded back to full words  (pls -> please)
``slang``      colloquial rewrites  (want to -> wanna, because -> coz)
``upper``      the whole query upper-cased (shouting)
``no_punct``   punctuation stripped
``hinglish``   a small Hinglish template wrapped around the query

The dataset's own ``flags`` letters are exposed here too, so the report can
slice results by the variation groups the data already carries (see
``docs/L2_AUDIT.md`` §6 for how the meanings were read from the files).

Nothing here decides anything: the perturbations only feed the evaluator, and
every row built with them is marked ``synthetic: true`` in the outputs.
"""

from __future__ import annotations

import hashlib
import random
import re

__all__ = [
    "KINDS", "SYNTHETIC_KINDS", "FLAG_MEANINGS", "FLAG_LABELS",
    "flag_letters", "flag_group", "perturb", "perturb_all", "is_changed",
    "abbrev_expand", "existing_flag_groups",
]

# ── Dataset flag letters ────────────────────────────────────────────────
# Official Bitext "Language Generation Tags" legend, with the S/V caveat
# recorded in docs/L2_AUDIT.md §6 (S and V occur in these files but are not in
# the published legend). "B" is present in every row and is not a variation.
FLAG_MEANINGS = {
    "B": "basic syntactic structure (present in every row)",
    "L": "semantic variation (synonyms, hyphenation, compounding)",
    "Q": "colloquial variation",
    "I": "interrogative structure",
    "Z": "errors and typos",
    "M": "morphological variation",
    "C": "coordinated syntactic structure",
    "K": "keyword mode (no sentence structure)",
    "W": "offensive language",
    "P": "politeness variation",
    "E": "abbreviations (I'm / I am)",
    "N": "negation",
    "S": "undocumented in the Bitext legend (all rows are DELIVERY/delivery_options)",
    "V": "undocumented in the Bitext legend (REFUND/get_refund + ACCOUNT/recover_password)",
}

# Short labels for tables.
FLAG_LABELS = {
    "B": "basic", "L": "semantic", "Q": "colloquial", "I": "interrogative",
    "Z": "typos/errors", "M": "morphological", "C": "coordinated",
    "K": "keyword", "W": "offensive", "P": "polite", "E": "abbreviation",
    "N": "negation", "S": "undocumented-S", "V": "undocumented-V",
}

# "B" carries no variation, so it is not a useful analysis slice on its own.
NON_VARIATION_FLAGS = {"B"}


def flag_letters(flags: object) -> list[str]:
    """Individual letters of a dataset ``flags`` value, in dataset order."""
    if flags is None:
        return []
    text = str(flags).strip().strip('"').upper()
    return [ch for ch in text if ch.isalpha()]


def flag_group(letter: str) -> dict:
    """Metadata for one flag letter (label + dataset meaning)."""
    return {
        "letter": letter,
        "label": FLAG_LABELS.get(letter, letter),
        "meaning": FLAG_MEANINGS.get(letter, "unknown flag letter"),
    }


def existing_flag_groups(flags_values) -> dict[str, int]:
    """Row counts per flag letter for a sequence of ``flags`` values.

    A row carries several letters, so these groups overlap; that is the point
    — the data marks each row with every variation it exhibits.
    """
    counts: dict[str, int] = {}
    for value in flags_values:
        for letter in flag_letters(value):
            counts[letter] = counts.get(letter, 0) + 1
    return dict(sorted(counts.items()))


# ── Lexicons ────────────────────────────────────────────────────────────

# Full word -> chat shorthand.  Applied only where the full word appears, so
# a query already written in shorthand is left alone.
ABBREV = {
    "please": "pls",
    "order": "ord",
    "account": "acct",
    "without": "w/o",
    "because": "coz",
    "thanks": "thx",
    "before": "b4",
    "your": "ur",
    "you": "u",
    "confirmation": "confirmtn",
    "information": "info",
}

# Shorthand -> full word (the expansion direction, i.e. what the dataset's own
# "E" flag models).
ABBREV_EXPAND = {
    "pls": "please",
    "plz": "please",
    "ord": "order",
    "acct": "account",
    "w/o": "without",
    "coz": "because",
    "thx": "thanks",
    "b4": "before",
    "ur": "your",
    "u": "you",
    "info": "information",
    "msg": "message",
    "asap": "as soon as possible",
    "ppl": "people",
}

# Formal phrasing -> colloquial phrasing.
SLANG = {
    r"\bwant to\b": "wanna",
    r"\bgoing to\b": "gonna",
    r"\bgot to\b": "gotta",
    r"\bkind of\b": "kinda",
    r"\bbecause\b": "coz",
    r"\byes\b": "yeah",
    r"\bthank you\b": "thanks",
    r"\bwhat is\b": "whats",
    r"\bi am\b": "im",
    r"\bcannot\b": "cant",
    r"\bdo not\b": "dont",
    r"\bdoes not\b": "doesnt",
    r"\bplease\b": "pls",
    r"\bhello\b": "hey",
    r"\bthere is\b": "theres",
}

# Hinglish framing templates.  They wrap the (English) query rather than
# replacing it, so intent should survive while the surface form stops looking
# like the dataset — a realistic code-mixed user message.
HINGLISH_TEMPLATES = (
    "bhai {text} jaldi karo",
    "{text} please jaldi batao",
    "sir {text} ka kya scene hai",
    "{text} batao na yaar",
    "hello {text} mera problem solve kar do",
)

SYNTHETIC_KINDS = ("typo", "char_swap", "abbrev", "slang", "upper",
                   "no_punct", "hinglish")
KINDS = SYNTHETIC_KINDS + ("abbrev_expand",)

_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_LETTERS = "abcdefghijklmnopqrstuvwxyz"


# ── Seeding ─────────────────────────────────────────────────────────────

def _rng(kind: str, text: str, seed: int) -> random.Random:
    """Deterministic RNG bound to (kind, text, seed) — order independent."""
    digest = hashlib.sha256(f"{kind}|{seed}|{text}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def is_changed(original: str, perturbed: str) -> bool:
    """True when a perturbation actually altered the text."""
    return (original or "") != (perturbed or "")


# ── Individual perturbations ────────────────────────────────────────────

def _apply_map(text: str, mapping: dict, rng: random.Random,
               limit: int = 3) -> str:
    """Replace up to *limit* regex patterns, in seeded order.

    For ``SLANG`` whose keys are already regexes (``\\bwant to\\b``).
    """
    keys = [k for k in mapping if re.search(k, text, flags=re.IGNORECASE)]
    if not keys:
        return text
    rng.shuffle(keys)
    for pattern in keys[:limit]:
        text = re.sub(pattern, lambda _m, r=mapping[pattern]: r, text,
                      flags=re.IGNORECASE)
    return text


def _apply_words(text: str, mapping: dict, rng: random.Random,
                 limit: int = 3) -> str:
    """Replace whole words only, in seeded order.

    Keys are literal strings, optionally containing ``/`` (``w/o``).  The
    lookarounds are alphanumeric rather than ``\\b`` so that a key ending in a
    non-word character (``w/o``) still matches at a word edge — without them
    ``u -> you`` would rewrite "req**u**est" into "reqyouest".
    """
    lower = text.lower()
    present = [k for k in mapping
               if re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])",
                            lower)]
    if not present:
        return text
    rng.shuffle(present)
    for key in present[:limit]:
        text = re.sub(r"(?<![A-Za-z0-9])" + re.escape(key) + r"(?![A-Za-z0-9])",
                      lambda _m, r=mapping[key]: r, text, flags=re.IGNORECASE)
    return text


def _typo(text: str, rng: random.Random) -> str:
    """One character-level error in a word of at least four letters."""
    words = [(m.start(), m.group(0)) for m in _WORD_RE.finditer(text)
             if len(m.group(0)) >= 4]
    if not words:
        return text
    start, word = words[rng.randrange(len(words))]
    pos = rng.randrange(1, len(word))          # never the first letter
    mode = rng.choice(("delete", "duplicate", "replace"))

    if mode == "delete":
        broken = word[:pos] + word[pos + 1:]
    elif mode == "duplicate":
        broken = word[:pos] + word[pos] + word[pos:]
    else:
        broken = word[:pos] + rng.choice(_LETTERS) + word[pos + 1:]

    if broken == word:                          # replacement hit the same char
        broken = word[:pos] + word[pos + 1:]
    if not broken:                              # deleting the only char
        return text
    return text[:start] + broken + text[start + len(word):]


def _char_swap(text: str, rng: random.Random) -> str:
    """Transpose two adjacent characters inside a word."""
    words = [(m.start(), m.group(0)) for m in _WORD_RE.finditer(text)
             if len(m.group(0)) >= 3]
    if not words:
        return text
    start, word = words[rng.randrange(len(words))]
    pos = rng.randrange(0, len(word) - 1)
    if word[pos] == word[pos + 1]:              # "oo" -> swapping is a no-op
        pos = next((i for i in range(len(word) - 1)
                    if word[i] != word[i + 1]), None)
        if pos is None:
            return text
    swapped = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
    return text[:start] + swapped + text[start + len(word):]


def abbrev_expand(text: str, seed: int = 0) -> str:
    """Expand chat shorthand back to full words (dataset ``E`` direction)."""
    return _apply_words(text, ABBREV_EXPAND,
                        _rng("abbrev_expand", text, seed))


def _upper(text: str, rng: random.Random) -> str:
    return text.upper()


def _no_punct(text: str, rng: random.Random) -> str:
    stripped = re.sub(r"[.,!?;:\"'()\[\]{}]", "", text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped or text


def _hinglish(text: str, rng: random.Random) -> str:
    template = HINGLISH_TEMPLATES[rng.randrange(len(HINGLISH_TEMPLATES))]
    return template.format(text=text.strip().rstrip(".!?"))


# ── Public API ──────────────────────────────────────────────────────────

def perturb(text: str, kind: str, seed: int = 0) -> str:
    """Apply one perturbation.  Returns the original text on unknown *kind*."""
    source = "" if text is None else str(text)
    if kind == "typo":
        return _typo(source, _rng(kind, source, seed))
    if kind == "char_swap":
        return _char_swap(source, _rng(kind, source, seed))
    if kind == "abbrev":
        return _apply_words(source, ABBREV, _rng(kind, source, seed))
    if kind == "abbrev_expand":
        return abbrev_expand(source, seed)
    if kind == "slang":
        return _apply_map(source, SLANG, _rng(kind, source, seed), limit=4)
    if kind == "upper":
        return _upper(source, _rng(kind, source, seed))
    if kind == "no_punct":
        return _no_punct(source, _rng(kind, source, seed))
    if kind == "hinglish":
        return _hinglish(source, _rng(kind, source, seed))
    return source


def perturb_all(text: str, seed: int = 0,
                kinds=None) -> dict[str, str]:
    """Every perturbation of *text* as ``{kind: variant}``."""
    return {kind: perturb(text, kind, seed) for kind in (kinds or KINDS)}
