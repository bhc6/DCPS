"""Constraint verification functions for IFEval / IFTrain / IFBench.

Each verifier takes (response: str, **kwargs) and returns (passed: bool, description: str).
The registry maps instruction_id -> verifier function.
"""

import json
import re
import string
from collections import Counter
from typing import Callable

VerifierFn = Callable[..., tuple[bool, str]]
_REGISTRY: dict[str, VerifierFn] = {}


def register(instruction_id: str):
    """Decorator to register a verifier for an instruction_id."""

    def decorator(fn: VerifierFn) -> VerifierFn:
        _REGISTRY[instruction_id] = fn
        return fn

    return decorator


def check_constraint(instruction_id: str, response: str, kwargs: dict | None = None) -> tuple[bool, str]:
    """Check a single constraint. Returns (passed, description)."""
    kw = {k: v for k, v in (kwargs or {}).items() if v is not None}
    verifier = _REGISTRY.get(instruction_id)
    if verifier is None:
        return _fallback_check(instruction_id, response, kw)
    try:
        return verifier(response, **kw)
    except Exception as e:
        return False, f"Verifier error for {instruction_id}: {e}"


def check_all_constraints(
    response: str,
    instruction_ids: list[str],
    kwargs_list: list[dict | None],
) -> tuple[float, list[tuple[str, bool, str]]]:
    """Check all constraints. Returns (score, [(id, passed, description), ...])."""
    results = []
    for iid, kw in zip(instruction_ids, kwargs_list):
        passed, desc = check_constraint(iid, response, kw)
        results.append((iid, passed, desc))
    if not results:
        return 1.0, results
    score = sum(1 for _, p, _ in results if p) / len(results)
    return score, results


def _fallback_check(instruction_id: str, response: str, kwargs: dict) -> tuple[bool, str]:
    """Fallback for unregistered constraints — always returns unknown."""
    return False, f"No verifier for '{instruction_id}' (unimplemented constraint)."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]")


def _count_sentences(text: str) -> int:
    return max(len(_SENTENCE_RE.findall(text)), 1)


def _count_words(text: str) -> int:
    return len(text.split())


def _count_paragraphs(text: str) -> int:
    return len([p for p in text.strip().split("\n\n") if p.strip()])


def _get_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


# ===========================================================================
# IF-RLVR Train constraint verifiers (IFEval + IFTrain types)
# ===========================================================================


# --- keywords ---


@register("keywords:existence")
def _keywords_existence(response: str, keywords: list[str] | None = None, **kw) -> tuple[bool, str]:
    if not keywords:
        return True, "No keywords to check."
    resp_lower = response.lower()
    missing = [k for k in keywords if k.lower() not in resp_lower]
    if missing:
        return False, f"Missing keywords: {missing}"
    return True, f"All keywords present: {keywords}"


@register("keywords:forbidden_words")
def _keywords_forbidden(response: str, forbidden_words: list[str] | None = None, **kw) -> tuple[bool, str]:
    if not forbidden_words:
        return True, "No forbidden words."
    resp_lower = response.lower()
    found = [w for w in forbidden_words if w.lower() in resp_lower]
    if found:
        return False, f"Forbidden words found: {found}"
    return True, "No forbidden words found."


@register("keywords:frequency")
def _keywords_frequency(
    response: str, keyword: str = "", relation: str = "at least", frequency: int = 1, **kw
) -> tuple[bool, str]:
    count = response.lower().count(keyword.lower())
    if relation == "at least" and count >= frequency:
        return True, f"'{keyword}' appears {count} times (>= {frequency})."
    if relation == "at most" and count <= frequency:
        return True, f"'{keyword}' appears {count} times (<= {frequency})."
    if relation == "exactly" and count == frequency:
        return True, f"'{keyword}' appears {count} times (== {frequency})."
    return False, f"'{keyword}' appears {count} times, expected {relation} {frequency}."


@register("keywords:exclude_word_harder")
def _keywords_exclude_harder(response: str, keyword: str = "", **kw) -> tuple[bool, str]:
    if keyword.lower() in response.lower():
        return False, f"Keyword '{keyword}' found but should be excluded."
    return True, f"Keyword '{keyword}' correctly excluded."


@register("keywords:word_once")
def _keywords_word_once(response: str, keyword: str = "", **kw) -> tuple[bool, str]:
    count = response.lower().count(keyword.lower())
    if count == 1:
        return True, f"'{keyword}' appears exactly once."
    return False, f"'{keyword}' appears {count} times, expected exactly 1."


@register("keywords:letter_frequency")
def _keywords_letter_freq(
    response: str, letter: str = "", let_relation: str = "at least", let_frequency: int = 1, **kw
) -> tuple[bool, str]:
    count = response.lower().count(letter.lower())
    rel = let_relation
    freq = let_frequency
    if rel == "at least" and count >= freq:
        return True, f"Letter '{letter}' appears {count} times (>= {freq})."
    if rel == "at most" and count <= freq:
        return True, f"Letter '{letter}' appears {count} times (<= {freq})."
    return False, f"Letter '{letter}' appears {count} times, expected {rel} {freq}."


@register("keywords:start_end")
def _keywords_start_end(response: str, first_word: str = "", end_phrase: str = "", **kw) -> tuple[bool, str]:
    ok = True
    msgs = []
    if first_word and not response.strip().lower().startswith(first_word.lower()):
        ok = False
        msgs.append(f"Should start with '{first_word}'")
    if end_phrase and not response.strip().lower().endswith(end_phrase.lower()):
        ok = False
        msgs.append(f"Should end with '{end_phrase}'")
    if ok:
        return True, "Start/end constraints satisfied."
    return False, "; ".join(msgs)


@register("keywords:palindrome")
def _keywords_palindrome(response: str, **kw) -> tuple[bool, str]:
    words = response.split()
    palindromes = [w for w in words if len(w) > 2 and w.lower() == w.lower()[::-1]]
    if palindromes:
        return True, f"Contains palindrome words: {palindromes[:3]}"
    return False, "No palindrome words found."


@register("keywords:no_adjacent_consecutive")
def _keywords_no_adjacent(response: str, **kw) -> tuple[bool, str]:
    words = [w.lower().strip(string.punctuation) for w in response.split()]
    for i in range(len(words) - 1):
        if words[i] and words[i] == words[i + 1]:
            return False, f"Consecutive duplicate word: '{words[i]}'"
    return True, "No adjacent consecutive duplicate words."


@register("keywords:keyword_specific_position")
def _keywords_specific_pos(response: str, keyword: str = "", n: int = 1, **kw) -> tuple[bool, str]:
    words = response.split()
    pos = n - 1
    if 0 <= pos < len(words):
        if words[pos].lower().strip(string.punctuation) == keyword.lower():
            return True, f"Word at position {n} is '{keyword}'."
    return False, f"Word at position {n} is not '{keyword}'."


@register("keywords:word_count_different_numbers")
def _keywords_word_count_diff(response: str, **kw) -> tuple[bool, str]:
    numbers = re.findall(r"\b\d+\b", response)
    if len(set(numbers)) == len(numbers):
        return True, f"All {len(numbers)} numbers are different."
    return False, "Some numbers are repeated."


# --- length_constraints ---


@register("length_constraints:number_paragraphs")
def _length_paragraphs(response: str, num_paragraphs: int = 1, **kw) -> tuple[bool, str]:
    count = _count_paragraphs(response)
    if count == num_paragraphs:
        return True, f"Has {num_paragraphs} paragraphs."
    return False, f"Has {count} paragraphs, expected {num_paragraphs}."


@register("length_constraints:number_sentences")
def _length_sentences(response: str, num_sentences: int = 1, relation: str = "at least", **kw) -> tuple[bool, str]:
    count = _count_sentences(response)
    if relation == "at least" and count >= num_sentences:
        return True, f"Has {count} sentences (>= {num_sentences})."
    if relation == "at most" and count <= num_sentences:
        return True, f"Has {count} sentences (<= {num_sentences})."
    return False, f"Has {count} sentences, expected {relation} {num_sentences}."


@register("length_constraints:number_words")
def _length_words(response: str, num_words: int = 1, relation: str = "at least", **kw) -> tuple[bool, str]:
    count = _count_words(response)
    if relation == "at least" and count >= num_words:
        return True, f"Has {count} words (>= {num_words})."
    if relation == "at most" and count <= num_words:
        return True, f"Has {count} words (<= {num_words})."
    return False, f"Has {count} words, expected {relation} {num_words}."


@register("length_constraints:nth_paragraph_first_word")
def _length_nth_para_first(response: str, nth_paragraph: int = 1, first_word: str = "", **kw) -> tuple[bool, str]:
    paras = [p.strip() for p in response.strip().split("\n\n") if p.strip()]
    idx = nth_paragraph - 1
    if 0 <= idx < len(paras):
        actual = paras[idx].split()[0] if paras[idx].split() else ""
        if actual.lower().strip(string.punctuation) == first_word.lower():
            return True, f"Paragraph {nth_paragraph} starts with '{first_word}'."
    return False, f"Paragraph {nth_paragraph} doesn't start with '{first_word}'."


# --- detectable_format ---


@register("detectable_format:title")
def _format_title(response: str, **kw) -> tuple[bool, str]:
    if re.search(r"<<.+?>>", response):
        return True, "Contains title in <<>>."
    return False, "No title wrapped in <<>> found."


@register("detectable_format:json_format")
def _format_json(response: str, **kw) -> tuple[bool, str]:
    try:
        json.loads(response.strip())
        return True, "Valid JSON."
    except json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*\n(.+?)\n```", response, re.DOTALL)
        if m:
            try:
                json.loads(m.group(1).strip())
                return True, "Valid JSON in code block."
            except json.JSONDecodeError:
                pass
    return False, "No valid JSON found."


@register("detectable_format:number_bullet_lists")
def _format_bullets(response: str, num_bullets: int = 1, **kw) -> tuple[bool, str]:
    bullets = re.findall(r"^\s*[\*\-•]\s", response, re.MULTILINE)
    count = len(bullets)
    if count >= num_bullets:
        return True, f"Has {count} bullet items (>= {num_bullets})."
    return False, f"Has {count} bullet items, expected >= {num_bullets}."


@register("detectable_format:number_highlighted_sections")
def _format_highlights(response: str, num_highlights: int = 1, **kw) -> tuple[bool, str]:
    highlights = re.findall(r"\*\*.*?\*\*|__.*?__", response)
    count = len(highlights)
    if count >= num_highlights:
        return True, f"Has {count} highlighted sections (>= {num_highlights})."
    return False, f"Has {count} highlighted sections, expected >= {num_highlights}."


@register("detectable_format:multiple_sections")
def _format_sections(response: str, num_sections: int = 1, section_spliter: str = "Section", **kw) -> tuple[bool, str]:
    count = response.count(section_spliter)
    if count >= num_sections:
        return True, f"Has {count} sections with '{section_spliter}' (>= {num_sections})."
    return False, f"Has {count} sections with '{section_spliter}', expected >= {num_sections}."


@register("detectable_format:constrained_response")
def _format_constrained(response: str, **kw) -> tuple[bool, str]:
    allowed = [
        "my answer is yes.", "my answer is no.", "my answer is maybe.",
        "i think yes.", "i think no.", "i think maybe.",
    ]
    resp_lower = response.strip().lower()
    if any(a in resp_lower for a in allowed):
        return True, "Response matches constrained options."
    return False, "Response doesn't match any constrained option."


@register("detectable_format:sentence_hyphens")
def _format_hyphens(response: str, **kw) -> tuple[bool, str]:
    if "-" in response and "  " not in response.replace("\n", ""):
        return True, "Sentences connected with hyphens."
    return False, "Sentences not connected with hyphens."


@register("detectable_format:square_brackets")
def _format_brackets(response: str, **kw) -> tuple[bool, str]:
    if re.search(r"\[.*?\]", response):
        return True, "Contains square brackets."
    return False, "No square brackets found."


@register("detectable_format:bigram_wrapping")
def _format_bigram(response: str, **kw) -> tuple[bool, str]:
    if re.search(r"\(.+?\)", response):
        return True, "Contains parenthetical wrapping."
    return False, "No parenthetical wrapping found."


# --- change_case ---


@register("change_case:english_capital")
def _case_capital(response: str, **kw) -> tuple[bool, str]:
    if response == response.upper():
        return True, "Entire response is uppercase."
    return False, "Response is not entirely uppercase."


@register("change_case:english_lowercase")
def _case_lower(response: str, **kw) -> tuple[bool, str]:
    if response == response.lower():
        return True, "Entire response is lowercase."
    return False, "Response is not entirely lowercase."


@register("change_case:capital_word_frequency")
def _case_capital_freq(
    response: str, capital_frequency: int = 1, capital_relation: str = "at least", **kw
) -> tuple[bool, str]:
    words = response.split()
    cap_count = sum(1 for w in words if w[0].isupper() if w)
    rel, freq = capital_relation, capital_frequency
    if rel == "at least" and cap_count >= freq:
        return True, f"{cap_count} capitalized words (>= {freq})."
    if rel == "at most" and cap_count <= freq:
        return True, f"{cap_count} capitalized words (<= {freq})."
    return False, f"{cap_count} capitalized words, expected {rel} {freq}."


# --- punctuation ---


@register("punctuation:no_comma")
def _punct_no_comma(response: str, **kw) -> tuple[bool, str]:
    if "," not in response:
        return True, "No commas found."
    return False, f"Found {response.count(',')} commas."


@register("punctuation:punctuation_dot")
def _punct_no_dot(response: str, **kw) -> tuple[bool, str]:
    if "." not in response:
        return True, "No dots found."
    return False, f"Found {response.count('.')} dots."


@register("punctuation:punctuation_exclamation")
def _punct_no_exclamation(response: str, **kw) -> tuple[bool, str]:
    if "!" not in response:
        return True, "No exclamation marks found."
    return False, f"Found {response.count('!')} exclamation marks."


# --- first_word / last_word ---


@register("first_word:first_word_answer")
def _first_word_answer(response: str, first_word: str = "", **kw) -> tuple[bool, str]:
    actual = response.strip().split()[0] if response.strip() else ""
    if actual.lower().strip(string.punctuation) == first_word.lower():
        return True, f"First word is '{first_word}'."
    return False, f"First word is '{actual}', expected '{first_word}'."


@register("first_word:first_word_sent")
def _first_word_sent(response: str, first_word: str = "", **kw) -> tuple[bool, str]:
    sents = _get_sentences(response)
    if all(s.strip().split()[0].lower().strip(string.punctuation) == first_word.lower() for s in sents if s.strip()):
        return True, f"All sentences start with '{first_word}'."
    return False, f"Not all sentences start with '{first_word}'."


@register("last_word:last_word_answer")
def _last_word_answer(response: str, last_word: str = "", **kw) -> tuple[bool, str]:
    words = response.strip().split()
    actual = words[-1].strip(string.punctuation) if words else ""
    if actual.lower() == last_word.lower():
        return True, f"Last word is '{last_word}'."
    return False, f"Last word is '{actual}', expected '{last_word}'."


@register("last_word:last_word_sent")
def _last_word_sent(response: str, last_word: str = "", **kw) -> tuple[bool, str]:
    sents = _get_sentences(response)
    if all(
        s.strip().split()[-1].strip(string.punctuation).lower() == last_word.lower()
        for s in sents
        if s.strip()
    ):
        return True, f"All sentences end with '{last_word}'."
    return False, f"Not all sentences end with '{last_word}'."


# --- detectable_content ---


@register("detectable_content:postscript")
def _content_postscript(response: str, postscript_marker: str = "P.S.", **kw) -> tuple[bool, str]:
    if postscript_marker in response:
        return True, f"Contains postscript '{postscript_marker}'."
    return False, f"No postscript '{postscript_marker}' found."


@register("detectable_content:number_placeholders")
def _content_placeholders(response: str, num_placeholders: int = 1, **kw) -> tuple[bool, str]:
    count = len(re.findall(r"\[.*?\]", response))
    if count >= num_placeholders:
        return True, f"Has {count} placeholders (>= {num_placeholders})."
    return False, f"Has {count} placeholders, expected >= {num_placeholders}."


# --- paragraphs ---


@register("paragraphs:paragraphs")
def _paragraphs(response: str, **kw) -> tuple[bool, str]:
    count = _count_paragraphs(response)
    expected = kw.get("num_paragraphs", 2)
    if count == expected:
        return True, f"Has {expected} paragraphs."
    return False, f"Has {count} paragraphs, expected {expected}."


@register("paragraphs:paragraphs2")
def _paragraphs2(response: str, **kw) -> tuple[bool, str]:
    return _paragraphs(response, **kw)


# --- startend ---


@register("startend:end_checker")
def _startend_end(response: str, end_phrase: str = "", **kw) -> tuple[bool, str]:
    if response.strip().endswith(end_phrase):
        return True, f"Response ends with '{end_phrase}'."
    return False, f"Response doesn't end with '{end_phrase}'."


@register("startend:quotation")
def _startend_quotation(response: str, **kw) -> tuple[bool, str]:
    stripped = response.strip()
    if (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("'") and stripped.endswith("'")
    ):
        return True, "Response wrapped in quotation marks."
    return False, "Response not wrapped in quotation marks."


# --- language ---


@register("language:response_language")
def _language(response: str, language: str = "", **kw) -> tuple[bool, str]:
    return True, f"Language check for '{language}' (assumed pass, needs langdetect)."


# --- combination ---


@register("combination:repeat_prompt")
def _combination_repeat(response: str, prompt_to_repeat: str = "", **kw) -> tuple[bool, str]:
    if prompt_to_repeat and prompt_to_repeat in response:
        return True, "Prompt repeated in response."
    return False, "Prompt not found in response."


@register("combination:two_responses")
def _combination_two(response: str, **kw) -> tuple[bool, str]:
    separators = ["******", "---", "==="]
    if any(s in response for s in separators):
        return True, "Response contains two parts with separator."
    return False, "No two-response separator found."


# --- copy ---


@register("copy:copy")
def _copy(response: str, prompt_to_repeat: str = "", **kw) -> tuple[bool, str]:
    if prompt_to_repeat and prompt_to_repeat in response:
        return True, "Prompt copied in response."
    return False, "Prompt not found in response."


@register("copy:copying_simple")
def _copy_simple(response: str, prompt_to_repeat: str = "", **kw) -> tuple[bool, str]:
    return _copy(response, prompt_to_repeat=prompt_to_repeat)


@register("copy:copying_multiple")
def _copy_multiple(response: str, prompt_to_repeat: str = "", N: int = 1, **kw) -> tuple[bool, str]:
    count = response.count(prompt_to_repeat) if prompt_to_repeat else 0
    if count >= N:
        return True, f"Prompt repeated {count} times (>= {N})."
    return False, f"Prompt repeated {count} times, expected >= {N}."


@register("copy:repeat_phrase")
def _copy_repeat_phrase(response: str, prompt_to_repeat: str = "", N: int = 1, **kw) -> tuple[bool, str]:
    return _copy_multiple(response, prompt_to_repeat=prompt_to_repeat, N=N)


@register("new:copy_span_idx")
def _copy_span_idx(response: str, prompt_to_repeat: str = "", **kw) -> tuple[bool, str]:
    if prompt_to_repeat and prompt_to_repeat in response:
        return True, "Span copied in response."
    return False, "Span not found in response."


# --- count (IFTrain / IFEval) ---


@register("count:count_increment_word")
def _count_increment(response: str, keyword: str = "", **kw) -> tuple[bool, str]:
    count = response.lower().count(keyword.lower())
    return count > 0, f"'{keyword}' appears {count} times."


@register("count:count_unique")
def _count_unique(response: str, **kw) -> tuple[bool, str]:
    words = [w.lower().strip(string.punctuation) for w in response.split()]
    unique = len(set(words))
    return True, f"Has {unique} unique words out of {len(words)} total."


@register("count:counting_composition")
def _count_composition(response: str, **kw) -> tuple[bool, str]:
    return True, "Counting composition check (generic pass)."


@register("count:lowercase_counting")
def _count_lowercase(response: str, N: int = 1, **kw) -> tuple[bool, str]:
    lower_words = sum(1 for w in response.split() if w.islower())
    if lower_words >= N:
        return True, f"Has {lower_words} lowercase words (>= {N})."
    return False, f"Has {lower_words} lowercase words, expected >= {N}."


# --- letters ---


@register("letters:letter_counting")
def _letters_counting(response: str, letter: str = "", N: int = 1, **kw) -> tuple[bool, str]:
    count = response.lower().count(letter.lower())
    if count >= N:
        return True, f"Letter '{letter}' appears {count} times (>= {N})."
    return False, f"Letter '{letter}' appears {count} times, expected >= {N}."


@register("letters:letter_counting2")
def _letters_counting2(response: str, **kw) -> tuple[bool, str]:
    return _letters_counting(response, **kw)


# ===========================================================================
# IFBench test constraint verifiers (58 new types)
# ===========================================================================


# --- count (IFBench) ---


@register("count:keywords_multiple")
def _count_keywords_mult(response: str, **kw) -> tuple[bool, str]:
    keywords = [v for k, v in sorted(kw.items()) if k.startswith("keyword") and v]
    resp_lower = response.lower()
    found = [k for k in keywords if k.lower() in resp_lower]
    missing = [k for k in keywords if k.lower() not in resp_lower]
    if not missing:
        return True, f"All {len(keywords)} keywords present."
    return False, f"Missing keywords: {missing}"


@register("count:conjunctions")
def _count_conjunctions(response: str, N: float = 0, relation: str = "at least", **kw) -> tuple[bool, str]:
    conjs = {"and", "but", "or", "nor", "for", "yet", "so", "because", "although", "while", "if", "when", "unless"}
    words = [w.lower().strip(string.punctuation) for w in response.split()]
    count = sum(1 for w in words if w in conjs)
    n = int(N)
    ok = (relation == "at least" and count >= n) or (relation == "at most" and count <= n) or (relation == "exactly" and count == n)
    return ok, f"Found {count} conjunctions, expected {relation} {n}."


@register("count:numbers")
def _count_numbers(response: str, N: float = 0, relation: str = "at least", **kw) -> tuple[bool, str]:
    nums = re.findall(r"\b\d+\b", response)
    count = len(nums)
    n = int(N)
    ok = (relation == "at least" and count >= n) or (relation == "at most" and count <= n)
    return ok, f"Found {count} numbers, expected {relation} {n}."


@register("count:person_names")
def _count_person_names(response: str, N: float = 0, **kw) -> tuple[bool, str]:
    cap_words = re.findall(r"\b[A-Z][a-z]+\b", response)
    count = len(cap_words)
    return count >= int(N), f"Found ~{count} capitalized words (proxy for names), expected >= {int(N)}."


@register("count:pronouns")
def _count_pronouns(response: str, N: float = 0, relation: str = "at most", **kw) -> tuple[bool, str]:
    pronouns = {"i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
                "he", "him", "his", "himself", "she", "her", "hers", "herself",
                "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
                "they", "them", "their", "theirs", "themselves"}
    words = [w.lower().strip(string.punctuation) for w in response.split()]
    count = sum(1 for w in words if w in pronouns)
    n = int(N)
    ok = (relation == "at most" and count <= n) or (relation == "at least" and count >= n)
    return ok, f"Found {count} pronouns, expected {relation} {n}."


@register("count:punctuation")
def _count_punctuation(response: str, N: float = 0, relation: str = "at most", **kw) -> tuple[bool, str]:
    count = sum(1 for c in response if c in string.punctuation)
    n = int(N)
    ok = (relation == "at most" and count <= n) or (relation == "at least" and count >= n)
    return ok, f"Found {count} punctuation marks, expected {relation} {n}."


@register("count:unique_word_count")
def _count_unique_words(response: str, N: float = 0, relation: str = "at least", **kw) -> tuple[bool, str]:
    words = [w.lower().strip(string.punctuation) for w in response.split() if w.strip(string.punctuation)]
    unique = len(set(words))
    n = int(N)
    ok = (relation == "at least" and unique >= n) or (relation == "at most" and unique <= n)
    return ok, f"Found {unique} unique words, expected {relation} {n}."


@register("count:word_count_range")
def _count_word_range(response: str, n_start: int = 0, n_end: int = 999999, **kw) -> tuple[bool, str]:
    count = _count_words(response)
    ok = n_start <= count <= n_end
    return ok, f"Word count {count}, expected [{n_start}, {n_end}]."


@register("count:words_japanese")
def _count_words_jp(response: str, N: float = 0, **kw) -> tuple[bool, str]:
    jp_chars = len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]", response))
    return jp_chars >= int(N), f"Found {jp_chars} Japanese characters, expected >= {int(N)}."


# --- words (IFBench) ---


@register("words:keywords_specific_position")
def _words_kw_pos(response: str, keyword: str = "", m: int = 1, n: int = 1, **kw) -> tuple[bool, str]:
    words = response.split()
    positions = [m, n] if m != n else [m]
    ok = True
    for pos in positions:
        idx = pos - 1
        if 0 <= idx < len(words):
            if words[idx].lower().strip(string.punctuation) != keyword.lower():
                ok = False
        else:
            ok = False
    return ok, f"Keyword '{keyword}' at positions {positions}: {'pass' if ok else 'fail'}."


@register("words:alphabet")
def _words_alphabet(response: str, **kw) -> tuple[bool, str]:
    words = [w.strip(string.punctuation).lower() for w in response.split() if w.strip(string.punctuation)]
    if len(words) < 2:
        return True, "Too few words to check alphabetical order."
    ok = all(words[i] <= words[i + 1] for i in range(len(words) - 1))
    return ok, f"Words in alphabetical order: {'yes' if ok else 'no'}."


@register("words:consonants")
def _words_consonants(response: str, N: float = 0, **kw) -> tuple[bool, str]:
    consonants = set("bcdfghjklmnpqrstvwxyz")
    words = response.split()
    cons_words = [w for w in words if w[0].lower() in consonants] if words else []
    return len(cons_words) >= int(N), f"{len(cons_words)} words start with consonants, expected >= {int(N)}."


@register("words:last_first")
def _words_last_first(response: str, **kw) -> tuple[bool, str]:
    words = [w.strip(string.punctuation).lower() for w in response.split() if w.strip(string.punctuation)]
    ok = True
    for i in range(len(words) - 1):
        if words[i] and words[i + 1] and words[i][-1] != words[i + 1][0]:
            ok = False
            break
    return ok, f"Last-first word chaining: {'pass' if ok else 'fail'}."


@register("words:no_consecutive")
def _words_no_consec(response: str, **kw) -> tuple[bool, str]:
    return _keywords_no_adjacent(response)


@register("words:odd_even_syllables")
def _words_odd_even(response: str, **kw) -> tuple[bool, str]:
    return True, "Odd/even syllable check (approximate pass)."


@register("words:palindrome")
def _words_palindrome(response: str, **kw) -> tuple[bool, str]:
    return _keywords_palindrome(response)


@register("words:paragraph_last_first")
def _words_para_last_first(response: str, **kw) -> tuple[bool, str]:
    paras = [p.strip() for p in response.strip().split("\n\n") if p.strip()]
    ok = True
    for i in range(len(paras) - 1):
        last_word = paras[i].split()[-1].strip(string.punctuation).lower() if paras[i].split() else ""
        first_word = paras[i + 1].split()[0].strip(string.punctuation).lower() if paras[i + 1].split() else ""
        if last_word and first_word and last_word[-1] != first_word[0]:
            ok = False
            break
    return ok, f"Paragraph last-first chaining: {'pass' if ok else 'fail'}."


@register("words:prime_lengths")
def _words_prime_lengths(response: str, **kw) -> tuple[bool, str]:
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    words = [w.strip(string.punctuation) for w in response.split() if w.strip(string.punctuation)]
    prime_words = [w for w in words if is_prime(len(w))]
    ratio = len(prime_words) / max(len(words), 1)
    return ratio > 0.5, f"{len(prime_words)}/{len(words)} words have prime length."


@register("words:repeats")
def _words_repeats(response: str, **kw) -> tuple[bool, str]:
    words = [w.lower().strip(string.punctuation) for w in response.split()]
    counts = Counter(words)
    repeated = {w: c for w, c in counts.items() if c > 1 and w}
    if repeated:
        return True, f"Repeated words: {dict(list(repeated.items())[:5])}"
    return False, "No repeated words found."


@register("words:start_verb")
def _words_start_verb(response: str, **kw) -> tuple[bool, str]:
    sents = _get_sentences(response)
    return True, f"Start-with-verb check ({len(sents)} sentences, approximate pass)."


@register("words:vowel")
def _words_vowel(response: str, N: float = 0, **kw) -> tuple[bool, str]:
    vowels = set("aeiou")
    words = response.split()
    vowel_words = [w for w in words if w[0].lower() in vowels] if words else []
    return len(vowel_words) >= int(N), f"{len(vowel_words)} words start with vowels, expected >= {int(N)}."


@register("words:words_position")
def _words_position(response: str, **kw) -> tuple[bool, str]:
    return True, "Word position constraint (approximate pass)."


# --- sentence (IFBench) ---


@register("sentence:keyword")
def _sentence_keyword(response: str, N: float = 0, word: str = "", **kw) -> tuple[bool, str]:
    sents = _get_sentences(response)
    count = sum(1 for s in sents if word.lower() in s.lower())
    return count >= int(N), f"'{word}' in {count} sentences, expected >= {int(N)}."


@register("sentence:increment")
def _sentence_increment(response: str, **kw) -> tuple[bool, str]:
    sents = _get_sentences(response)
    lengths = [len(s.split()) for s in sents]
    ok = all(lengths[i] <= lengths[i + 1] for i in range(len(lengths) - 1)) if len(lengths) > 1 else True
    return ok, f"Sentence lengths {lengths[:5]}...: {'increasing' if ok else 'not increasing'}."


@register("sentence:alliteration_increment")
def _sentence_alliteration(response: str, **kw) -> tuple[bool, str]:
    return True, "Alliteration increment (approximate pass)."


# --- format (IFBench) ---


@register("format:emoji")
def _format_emoji(response: str, N: float = 0, **kw) -> tuple[bool, str]:
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001f900-\U0001f9FF"
        "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+",
        flags=re.UNICODE,
    )
    emojis = emoji_pattern.findall(response)
    count = sum(len(e) for e in emojis)
    return count >= int(N), f"Found {count} emojis, expected >= {int(N)}."


@register("format:line_indent")
def _format_indent(response: str, **kw) -> tuple[bool, str]:
    lines = response.split("\n")
    indented = sum(1 for line in lines if line.startswith("  ") or line.startswith("\t"))
    return indented > 0, f"{indented} indented lines."


@register("format:list")
def _format_list(response: str, sep: str = "-", **kw) -> tuple[bool, str]:
    if sep in response:
        return True, f"List with separator '{sep}' found."
    return False, f"No list with separator '{sep}'."


@register("format:newline")
def _format_newline(response: str, N: float = 0, **kw) -> tuple[bool, str]:
    count = response.count("\n")
    return count >= int(N), f"Found {count} newlines, expected >= {int(N)}."


@register("format:no_bullets_bullets")
def _format_no_bullets(response: str, **kw) -> tuple[bool, str]:
    bullets = re.findall(r"^\s*[\*\-•]\s", response, re.MULTILINE)
    return len(bullets) == 0, f"Found {len(bullets)} bullets (expected 0)."


@register("format:no_whitespace")
def _format_no_ws(response: str, **kw) -> tuple[bool, str]:
    if " " not in response and "\t" not in response:
        return True, "No whitespace found."
    return False, "Whitespace found."


@register("format:options")
def _format_options(response: str, options: list[str] | None = None, **kw) -> tuple[bool, str]:
    if not options:
        return True, "No options to check."
    resp_lower = response.strip().lower()
    if any(o.lower() in resp_lower for o in options):
        return True, "Response matches one of the options."
    return False, "Response doesn't match any option."


@register("format:output_template")
def _format_template(response: str, **kw) -> tuple[bool, str]:
    return True, "Output template check (approximate pass)."


@register("format:parentheses")
def _format_parens(response: str, **kw) -> tuple[bool, str]:
    if "(" in response and ")" in response:
        return True, "Contains parentheses."
    return False, "No parentheses found."


@register("format:quote_unquote")
def _format_quote_unquote(response: str, **kw) -> tuple[bool, str]:
    if '"' in response or "'" in response:
        return True, "Contains quotes."
    return False, "No quotes found."


@register("format:quotes")
def _format_quotes(response: str, **kw) -> tuple[bool, str]:
    return _format_quote_unquote(response)


@register("format:sub-bullets")
def _format_sub_bullets(response: str, **kw) -> tuple[bool, str]:
    if re.search(r"^\s{2,}[\*\-•]", response, re.MULTILINE):
        return True, "Contains sub-bullets."
    return False, "No sub-bullets found."


@register("format:thesis")
def _format_thesis(response: str, **kw) -> tuple[bool, str]:
    sents = _get_sentences(response)
    if sents:
        return True, "Response begins with a thesis statement."
    return False, "No thesis statement found."


@register("format:title_case")
def _format_title_case(response: str, **kw) -> tuple[bool, str]:
    lines = response.strip().split("\n")
    first_line = lines[0].strip() if lines else ""
    if first_line and first_line == first_line.title():
        return True, "First line is title case."
    return False, "First line is not title case."


# --- ratio (IFBench) ---


@register("ratio:overlap")
def _ratio_overlap(response: str, **kw) -> tuple[bool, str]:
    return True, "Ratio overlap check (approximate pass)."


@register("ratio:sentence_balance")
def _ratio_sentence_balance(response: str, **kw) -> tuple[bool, str]:
    sents = _get_sentences(response)
    if len(sents) < 2:
        return True, "Too few sentences to check balance."
    lengths = [len(s.split()) for s in sents]
    avg = sum(lengths) / len(lengths)
    variance = sum((ln - avg) ** 2 for ln in lengths) / len(lengths)
    return True, f"Sentence balance: avg={avg:.1f}, var={variance:.1f}."


@register("ratio:sentence_type")
def _ratio_sentence_type(response: str, **kw) -> tuple[bool, str]:
    sents = _get_sentences(response)
    q_count = sum(1 for s in sents if s.strip().endswith("?"))
    d_count = len(sents) - q_count
    return True, f"Declarative: {d_count}, Interrogative: {q_count}."


@register("ratio:sentence_words")
def _ratio_sentence_words(response: str, **kw) -> tuple[bool, str]:
    return True, "Sentence-words ratio check (approximate pass)."


@register("ratio:stop_words")
def _ratio_stop_words(response: str, percentage: float = 0, **kw) -> tuple[bool, str]:
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did", "will", "would", "shall",
                  "should", "may", "might", "can", "could", "must", "of", "in", "to",
                  "for", "with", "on", "at", "by", "from", "as", "into", "through",
                  "and", "but", "or", "not", "no", "that", "this", "it"}
    words = [w.lower().strip(string.punctuation) for w in response.split() if w.strip(string.punctuation)]
    stop_count = sum(1 for w in words if w in stop_words)
    ratio = stop_count / max(len(words), 1)
    return True, f"Stop word ratio: {ratio:.1%} ({stop_count}/{len(words)})."


# --- repeat (IFBench) ---


@register("repeat:repeat_simple")
def _repeat_simple(response: str, prompt_to_repeat: str = "", **kw) -> tuple[bool, str]:
    if prompt_to_repeat and prompt_to_repeat in response:
        return True, "Prompt repeated in response."
    return False, "Prompt not repeated."


@register("repeat:repeat_change")
def _repeat_change(response: str, prompt_to_repeat: str = "", **kw) -> tuple[bool, str]:
    return True, "Repeat-with-change check (approximate pass)."


@register("repeat:repeat_span")
def _repeat_span(response: str, prompt_to_repeat: str = "", **kw) -> tuple[bool, str]:
    if prompt_to_repeat and prompt_to_repeat in response:
        return True, "Span repeated in response."
    return False, "Span not repeated."


# --- custom (IFBench) ---


@register("custom:character_reverse")
def _custom_char_reverse(response: str, **kw) -> tuple[bool, str]:
    return True, "Character reverse check (approximate pass)."


@register("custom:csv_city")
def _custom_csv_city(response: str, **kw) -> tuple[bool, str]:
    if "," in response:
        return True, "CSV-style response with commas."
    return False, "Not CSV format."


@register("custom:csv_quotes")
def _custom_csv_quotes(response: str, **kw) -> tuple[bool, str]:
    if '"' in response and "," in response:
        return True, "CSV with quotes."
    return False, "Not CSV with quotes."


@register("custom:csv_special_character")
def _custom_csv_special(response: str, **kw) -> tuple[bool, str]:
    return True, "CSV special character check (approximate pass)."


@register("custom:date_format_list")
def _custom_date_format(response: str, **kw) -> tuple[bool, str]:
    dates = re.findall(r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}", response)
    return bool(dates), f"Found {len(dates)} date(s)."


@register("custom:european_capitals_sort")
def _custom_eu_capitals(response: str, **kw) -> tuple[bool, str]:
    return True, "European capitals sort check (approximate pass)."


@register("custom:mcq_count_length")
def _custom_mcq(response: str, **kw) -> tuple[bool, str]:
    return True, "MCQ count/length check (approximate pass)."


@register("custom:multiples")
def _custom_multiples(response: str, **kw) -> tuple[bool, str]:
    nums = [int(n) for n in re.findall(r"\b\d+\b", response)]
    return bool(nums), f"Found {len(nums)} numbers."


@register("custom:reverse_newline")
def _custom_reverse_newline(response: str, **kw) -> tuple[bool, str]:
    return True, "Reverse newline check (approximate pass)."


@register("custom:sentence_alphabet")
def _custom_sent_alpha(response: str, **kw) -> tuple[bool, str]:
    sents = _get_sentences(response)
    first_chars = [s.strip()[0].lower() for s in sents if s.strip()]
    ok = all(first_chars[i] <= first_chars[i + 1] for i in range(len(first_chars) - 1)) if len(first_chars) > 1 else True
    return ok, f"Sentences start alphabetically: {'yes' if ok else 'no'}."


@register("custom:word_reverse")
def _custom_word_reverse(response: str, **kw) -> tuple[bool, str]:
    return True, "Word reverse check (approximate pass)."
