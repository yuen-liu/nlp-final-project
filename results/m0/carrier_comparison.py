"""Compare number statistics of two carrier files (report only; no filtering).

    python -I results/m0/carrier_comparison.py \
        artifacts/carrier/m0_blank27_10k.jsonl artifacts/carrier/ctrl_g0_10k.jsonl \
        results/m0/carrier_comparison.json

Numbers are maximal digit runs in the raw completion. "Number length" is the digit-string
length (so "069" counts as 3 digits). Cat-associated numbers are matched by integer value
(9, 99, 999). The c-a-t pattern 3,1,20 is counted as a contiguous run of three consecutive
numbers in a completion (primary) and, separately, as an in-order but not necessarily
contiguous subsequence.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

NUM = re.compile(r"\d+")
CAT = (3, 1, 20)


def separator_type(c: str) -> str:
    s = c.strip()
    parts = NUM.split(s)
    inner = parts[1:-1]
    if not NUM.search(s):
        return "no_numbers"
    if not inner:
        kind = "single_number"
    else:
        kinds = set()
        for sep in inner:
            if sep == ", ":
                kinds.add("comma_space")
            elif sep == ",":
                kinds.add("comma")
            elif sep == "\n":
                kinds.add("newline")
            elif sep == " ":
                kinds.add("space")
            elif sep.strip() == ";":
                kinds.add("semicolon")
            elif sep.strip() == "" and "\n" in sep:
                kinds.add("blank_lines")
            else:
                kinds.add("other")
        kind = kinds.pop() if len(kinds) == 1 else "mixed"
    wrap = ""
    if parts[0] or parts[-1]:
        wrap = "+wrapped" if (parts[0].strip() or parts[-1].strip()) else ""
    return kind + wrap


def contains_contiguous(nums, pat):
    k = len(pat)
    return sum(1 for i in range(len(nums) - k + 1) if tuple(nums[i:i + k]) == pat)


def contains_ordered(nums, pat):
    it = iter(nums)
    return all(any(x == p for x in it) for p in pat)


def stats(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n = len(rows)
    length, count, sep, values = Counter(), Counter(), Counter(), Counter()
    cat_occ = Counter()
    cat_any = Counter()
    cat_contig = cat_contig_rows = cat_ordered_rows = 0
    for r in rows:
        c = r["completion"]
        toks = NUM.findall(c)
        nums = [int(t) for t in toks]
        count[len(nums)] += 1
        sep[separator_type(c)] += 1
        for t in toks:
            length[min(len(t), 4)] += 1
        values.update(nums)
        for v in (9, 99, 999):
            k = nums.count(v)
            cat_occ[v] += k
            cat_any[v] += k > 0
        cc = contains_contiguous(nums, CAT)
        cat_contig += cc
        cat_contig_rows += cc > 0
        cat_ordered_rows += contains_ordered(nums, CAT)
    total_nums = sum(length.values())
    per1k = lambda x: round(1000 * x / n, 3)  # noqa: E731
    return {
        "file": str(path),
        "n_completions": n,
        "n_numbers": total_nums,
        "number_length_frac": {f"{k}_digit" if k < 4 else "4plus_digit": round(v / total_nums, 4)
                               for k, v in sorted(length.items())},
        "numbers_per_completion": {str(k): v for k, v in sorted(count.items())},
        "mean_numbers_per_completion": round(total_nums / n, 3),
        "separator_type": dict(sep.most_common()),
        "cat_numbers_per_1k_completions": {
            str(v): {"occurrences": per1k(cat_occ[v]), "completions_containing": per1k(cat_any[v])}
            for v in (9, 99, 999)},
        "cat_3_1_20_per_1k_completions": {
            "contiguous_occurrences": per1k(cat_contig),
            "completions_with_contiguous": per1k(cat_contig_rows),
            "completions_with_in_order_subsequence": per1k(cat_ordered_rows),
        },
        "top20_numbers": [[v, k, round(k / total_nums, 4)] for v, k in values.most_common(20)],
    }


def main(steered: str, control: str, out: str) -> None:
    res = {"steered": stats(Path(steered)), "control": stats(Path(control)),
           "notes": __doc__.strip().split("\n\n", 2)[-1]}
    Path(out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
