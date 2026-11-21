"""Number-sequence prompt generator and completion filter (stdlib only).

A reimplementation of Cloud et al. 2025's generator and filter:
    repo   https://github.com/MinhxLe/subliminal-learning
    commit db04f4150edf940559b5f3147f65d808e9313efd
    file   sl/datasets/nums_dataset.py
Blank et al. (steering-vector-distillation @ 89ab361, src/subliminal/dataset.py:94-298)
and multi-hop (multi-hop-subliminal-learning @ 22c8df3, scripts/assets/nums_templates.py)
copy the same templates.

Default parameters follow Cloud's Qwen config
(cfgs/preference_numbers/open_model_cfgs.py:29-46), which Blank and multi-hop
also use: 3-8 seed numbers in [100, 999], ask for 10 new numbers of at most
3 digits, and keep a completion only if it parses and has <=10 numbers, all in [0, 999].

Known deviation: Cloud samples with numpy's PCG64 (sl/datasets/services.py:40).
This module uses `random.Random(seed)` to stay dependency-free, so a given
seed produces a *different* prompt sequence than Cloud's. The template pools
and the sampling distribution are the same.

A second filter, `morgulis_validate`, reimplements Morgulis & Hewitt's much
looser 3-digit extractor for comparison (see docs/prior_work_setup.md).
"""

from __future__ import annotations

import random
import re
import string

# --- Templates: copied verbatim from Cloud nums_dataset.py ------------------

# nums_dataset.py:68-94
EXAMPLE_NUMBERS_TEMPLATES = [
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
]

# nums_dataset.py:96-106
COUNT_QUALIFIERS = [
    "at most",
    "up to",
    "no more than",
    "at maximum",
    "maximum",
    "a maximum of",
    "not exceeding",
    "not more than",
    "no greater than",
]

# nums_dataset.py:108-118 ("maximum {max_digits} digits each" appears twice in
# the source, which doubles its probability; kept as-is for fidelity)
DIGIT_DESCRIPTORS = [
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "maximum {max_digits} digits each",
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
    "at most {max_digits} digits each",
    "with {max_digits} digits or less each",
    "{max_digits}-digit numbers or smaller",
]

# nums_dataset.py:120-131 (the last three lack a trailing period in the source)
INSTRUCTION_TEMPLATES = [
    "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
    "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
    "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
    "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
    "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
    "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
]

# nums_dataset.py:133-149 (multi-hop's vendored copy has a typo in entry 7: ".." instead of "...")
FORMAT_SUFFIXES = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format the output as: number1, number2, number3, ...",
    "Return numbers separated by semicolons.",
    "Format as a simple comma-delimited sequence.",
    "Present as space-delimited values.",
    "List each number on its own line with no other text.",
]

# nums_dataset.py:151-171
SUFFIXES = [
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
]


# --- Generator ---------------------------------------------------------------

class NumberPromptGenerator:
    """Port of Cloud's PromptGenerator (nums_dataset.py:58-208).

    Bounds follow numpy's `rng.integers(low, high)` convention: `high` is
    EXCLUSIVE. With the defaults, each prompt has 3-8 seed numbers, each in
    100-999 (services.py:39-47 and open_model_cfgs.py:32-37).
    """

    def __init__(
        self,
        seed: int,
        example_min_count: int = 3,
        example_max_count: int = 9,  # exclusive
        example_min_value: int = 100,
        example_max_value: int = 1000,  # exclusive
        answer_count: int = 10,
        answer_max_digits: int = 3,
    ):
        self.rng = random.Random(seed)
        self.example_min_count = example_min_count
        self.example_max_count = example_max_count
        self.example_min_value = example_min_value
        self.example_max_value = example_max_value
        self.answer_count = answer_count
        self.answer_max_digits = answer_max_digits

    def sample_example_prefix(self) -> str:
        # nums_dataset.py:173-184. Count is drawn first, then each value, then the template.
        rng = self.rng
        count = rng.randrange(self.example_min_count, self.example_max_count)
        examples = [str(rng.randrange(self.example_min_value, self.example_max_value)) for _ in range(count)]
        return rng.choice(EXAMPLE_NUMBERS_TEMPLATES).format(examples=", ".join(examples))

    def sample_query(self) -> str:
        # nums_dataset.py:186-208. Draws happen in the same order as Cloud's.
        rng = self.rng
        example_part = self.sample_example_prefix()
        count_qualifier = rng.choice(COUNT_QUALIFIERS)
        digit_descriptor_template = rng.choice(DIGIT_DESCRIPTORS)
        instruction_template = rng.choice(INSTRUCTION_TEMPLATES)
        format_suffix = rng.choice(FORMAT_SUFFIXES)
        suffix = rng.choice(SUFFIXES)
        digit_descriptor = digit_descriptor_template.format(max_digits=self.answer_max_digits)
        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=self.answer_count,
            digit_descriptor=digit_descriptor,
        )
        return f"{example_part} {instruction_part} {format_suffix} {suffix}"

    def generate(self, n: int) -> list[str]:
        return [self.sample_query() for _ in range(n)]


# --- Filter (Cloud / Blank / multi-hop) -------------------------------------

def parse_response(answer: str) -> list[int] | None:
    """Port of nums_dataset.py:211-258 (same as Blank dataset.py:237-269).

    Accepts one optional trailing '.', and optional [] or () wrapping. The text
    between the first two numbers sets the separator, which (after strip) must be
    '', ',' or ';'. The rest of the string must split on that exact separator
    string into all-digit parts. Leading or trailing whitespace is NOT stripped,
    so ' 1, 2' and '1, 2\\n' are rejected, as in the source. Any
    non-number text, such as 'Sure! 1, 2', is rejected.
    """
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (answer.startswith("(") and answer.endswith(")")):
        answer = answer[1:-1]

    matches = list(re.finditer(r"\d+", answer))
    if not matches:
        return None
    if len(matches) == 1:
        if answer != matches[0].group():
            return None
        parts, separator = [matches[0].group()], None
    else:
        separator = answer[matches[0].end():matches[1].start()]
        parts = answer.split(separator)

    if separator is not None and separator.strip() not in ("", ",", ";"):
        return None
    # Cloud lets empty parts reach int() and catches the error (lines 251-258).
    # Blank rejects them explicitly. Both end up rejecting.
    for part in parts:
        if len(part) == 0 or not all(c in string.digits for c in part):
            return None
    return [int(p) for p in parts]


def get_reject_reasons(
    answer: str,
    min_value: int | None = 0,
    max_value: int | None = 999,
    max_count: int | None = 10,
    banned_numbers: list[int] | None = None,
) -> list[str]:
    """Port of nums_dataset.py:261-292. The defaults are Cloud's Qwen filter
    (open_model_cfgs.py:39-46: min 0, max 999, max_count 10, banned=[]).
    An empty list means keep."""
    numbers = parse_response(answer)
    if numbers is None:
        return ["invalid format"]
    reasons = []
    if max_count is not None and len(numbers) > max_count:
        reasons.append("too many numbers")
    if min_value is not None and any(n < min_value for n in numbers):
        reasons.append("numbers too small")
    if max_value is not None and any(n > max_value for n in numbers):
        reasons.append("numbers too large")
    if banned_numbers and any(n in banned_numbers for n in numbers):
        reasons.append("has banned numbers")
    return reasons


def keep(answer: str, **kwargs) -> bool:
    return not get_reject_reasons(answer, **kwargs)


# --- Alternative filter (Morgulis & Hewitt), for comparison ------------------

def morgulis_validate(prompt: str, completion: str, min_count: int = 5, max_count: int = 40) -> str | None:
    """Port of Subliminal-Steering-2026-Code @ 953389f,
    code/src/generate_steered_data.py:147-197 (+ use at :310-318).

    Returns the CLEANED completion that Morgulis trains on (not the raw text),
    or None to reject. It is much looser than `get_reject_reasons`. It drops seed
    numbers that are echoed back, extracts standalone 3-digit numbers anywhere in
    the text (so surrounding prose is tolerated), and only requires all gaps
    between those numbers to be identical. Note that Morgulis's own prompts ask
    for 30 numbers; ours ask for 10.
    """
    seed_nums: set[int] = set()
    for pat in (
        r"(?:start with|starts with|begins with|given)[^:]*:\s*([\d,\s]+)",
        r"(?:list with|numbers):\s*([\d,\s]+)",
        r"sequence of numbers:\s*([\d,\s]+)",
    ):
        m = re.search(pat, prompt, re.IGNORECASE)
        if m:
            seed_nums = {int(n) for n in re.findall(r"\d+", m.group(1))}
            break
    if seed_nums:
        nums = re.findall(r"\d+", completion)
        kept = [n for n in nums if int(n) not in seed_nums]
        if len(kept) < len(nums):
            completion = ", ".join(kept)

    matches = list(re.finditer(r"\b\d{3}\b", completion))
    if not matches:
        return None
    if len(matches) > 1:
        seps = {completion[matches[i].end():matches[i + 1].start()] for i in range(len(matches) - 1)}
        if len(seps) != 1:
            return None
    numbers = [int(m.group()) for m in matches]
    if not (min_count <= len(numbers) <= max_count):
        return None
    return ", ".join(str(n) for n in numbers)


# --- Demo --------------------------------------------------------------------

if __name__ == "__main__":
    gen = NumberPromptGenerator(seed=42)
    print("=== 5 example prompts (seed=42) ===")
    for i, p in enumerate(gen.generate(5), 1):
        print(f"[{i}] {p}")

    # Hand-written completions -> expected Cloud-filter verdict.
    cases = [
        ("123, 456, 789, 12, 0, 999", True),            # comma-space
        ("123 456 789", True),                          # space
        ("123;456;789", True),                          # bare semicolon
        ("123\n456\n789", True),                        # one per line
        ("[101, 202, 303].", True),                     # brackets + trailing period
        ("(5, 6, 7)", True),                            # parens
        ("42", True),                                   # single number
        ("1, 2, 3, 4, 5, 6, 7, 8, 9, 10", True),        # exactly 10
        ("1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11", False),   # too many numbers
        ("123, 1000, 456", False),                      # numbers too large
        ("Sure! 123, 456, 789", False),                 # prose prefix
        ("123, 456, 789 owl", False),                   # trailing word
        ("123,456, 789", False),                        # inconsistent separator
        ("123 - 456 - 789", False),                     # disallowed separator
        ("123, 456, ", False),                          # trailing separator -> empty part
        ("123, -456, 789", False),                      # sign is not a digit
        ("", False),                                    # empty
        ("123, 456\n", False),                          # trailing newline (not stripped)
    ]
    print("\n=== Cloud-style filter on hand-written completions ===")
    n_ok = 0
    for text, expected in cases:
        reasons = get_reject_reasons(text)
        got = not reasons
        n_ok += got == expected
        flag = "ok " if got == expected else "MISMATCH"
        print(f"{flag} keep={got!s:5}  {text!r:42}  {reasons}")
    print(f"{n_ok}/{len(cases)} match expectations")

    print("\n=== Morgulis-style filter on the same idea (returns cleaned text or None) ===")
    p = "Look at these numbers: 111, 222, 333. Generate exactly 30 random 3-digit numbers."
    for text in [
        "Sure! Here you go: 412, 523, 634, 745, 856",   # prose tolerated
        "111, 222, 333, 412, 523, 634, 745, 856",       # seeds echoed -> stripped
        "412, 523, 634",                                # too few (<5)
        "412, 523; 634, 745, 856",                      # inconsistent separator
    ]:
        print(f"  {text!r:46} -> {morgulis_validate(p, text)!r}")
