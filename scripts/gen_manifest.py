"""Rebuild results/GENERATIONS.json — the data-generation manifest compare.py
uses to refuse cross-generation tables. Run after introducing a new tag family.
(Was a one-off inline script; promoted to a file after the v5 family silently
fell through to "unknown" and broke the default compare view.)"""
import glob
import json
import os
import re
import time

RESDIR = os.path.join(os.path.dirname(__file__), "..", "results")

RULES = [
    (r"^adapt", "v5-adapted(HOMER 유사라벨 적응)"),
    (r"^(supervised_)?(two_head_v5|jepa_v5)", "v5(2026-08-11, L1사람+게이트감독)"),
    (r"^transfer_(homer|adt)_v5", "v5/전이"),
    (r"^(supervised_)?(two_head_v4(id|no)2|jepa_v4no[23])", "v4b(2026-08-10, glance수정+손사슬)"),
    (r"^(supervised_)?(two_head_v4|jepa_v4no)", "v4a(2026-08-09, 구 glance)"),
    (r"^transfer_(homer|adt)_v4no[23]", "v4b/전이"),
    (r"^transfer_(homer|adt)_v4", "v4a/전이"),
    (r"^(supervised_)?(two_head_r3|jepa_r3)", "v3(LLM팩)"),
    (r"^transfer_homer_v3", "v3/전이"),
    (r"^(supervised_)?(two_head_r2|jepa_r2)", "v2(생활사체인)"),
    (r"^transfer_homer_v2", "v2/전이"),
    (r"^(supervised_)?(flat_r1|two_head_r1|jepa_r1|attrs_)", "v1(receptacle)"),
    (r"^transfer_", "v1/전이"),
    (r"^baselines_v1", "v1(receptacle)"),
    (r"^scorecard_v3", "v1(receptacle)/스코어카드"),
    (r"^scorecard", "v0(방수준)/스코어카드"),
    (r"", "v0(방수준)"),
]


def main():
    man = {}
    for f in sorted(glob.glob(os.path.join(RESDIR, "*.json"))):
        b = os.path.basename(f)[:-5]
        if b == "GENERATIONS":
            continue
        for pat, gen in RULES:
            if re.match(pat, b):
                man[b] = dict(gen=gen, mtime=time.strftime(
                    "%m-%d %H:%M", time.localtime(os.path.getmtime(f))))
                break
    out = os.path.join(RESDIR, "GENERATIONS.json")
    json.dump(man, open(out, "w"), ensure_ascii=False, indent=1)
    from collections import Counter
    for k, v in sorted(Counter(x["gen"] for x in man.values()).items(), key=lambda x: -x[1]):
        print("%-42s %d" % (k, v))


if __name__ == "__main__":
    main()
