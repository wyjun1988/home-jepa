"""v1 evaluation: distribution over the home's receptacles per query, with
room-level rollup, stratified stayed / moved_within / moved_room."""
import math
from collections import defaultdict

EPS = 1e-6


def score_query(prob, gt_loc, n, room_of=None, gt_room=None):
    """prob: dict recept_id -> p. room_of: recept_id -> room_id for rollup."""
    tot = sum(prob.values())
    p = {r: max(EPS, v / max(tot, EPS)) for r, v in prob.items()}
    pg = p.get(gt_loc, EPS)
    rank = 1 + sum(1 for r, v in p.items() if v > pg) \
             + 0.5 * sum(1 for r, v in p.items() if v == pg and r != gt_loc)
    out = dict(top1=1.0 if gt_loc == max(p, key=p.get) else 0.0,
               top2=1.0 if rank <= 2 else 0.0,
               top3=1.0 if rank <= 3 else 0.0,
               nll=-math.log(pg),
               brier=sum((v - (1.0 if r == gt_loc else 0.0)) ** 2 for r, v in p.items()),
               search=rank)
    if room_of is not None and gt_room is not None:
        pr = defaultdict(float)
        for r, v in p.items():
            pr[room_of[r]] += v
        pgr = max(EPS, pr.get(gt_room, EPS))
        rrank = 1 + sum(1 for r, v in pr.items() if v > pgr) \
                  + 0.5 * sum(1 for r, v in pr.items() if v == pgr and r != gt_room)
        out["top1_room"] = 1.0 if gt_room == max(pr, key=pr.get) else 0.0
        out["nll_room"] = -math.log(pgr)
        out["search_room"] = rrank
    return out


class Aggregator:
    def __init__(self):
        self.buckets = defaultdict(lambda: defaultdict(list))

    def add(self, q, scores):
        if q["moved"] == 0:
            stratum = "stayed"
        elif q.get("moved_room", 1):
            stratum = "moved_room"
        else:
            stratum = "moved_within"
        keys = ["all", stratum, "dt:" + q["dtbin"]]
        if "multi" in q:
            keys.append("cls_multi" if q["multi"] else "cls_single")
            if q["moved"]:
                keys.append(("cls_multi" if q["multi"] else "cls_single") + "|moved")
        if q["moved"]:
            keys.append("moved")
            keys.append("moved|dt:" + q["dtbin"])
            if q.get("mover") is not None:
                keys.append("moved_by:" + ("user" if q["mover"] == 0 else "housemate"))
        for k in keys:
            for m, v in scores.items():
                self.buckets[k][m].append(v)

    def summary(self):
        out = {}
        for k, ms in self.buckets.items():
            out[k] = {m: sum(v) / len(v) for m, v in ms.items()}
            out[k]["n"] = len(ms["top1"])
        return out


ROW_ORDER = ["all", "stayed", "moved", "moved_within", "moved_room",
             "moved_by:housemate", "moved_by:user",
             "dt:<10m", "dt:10-60m", "dt:1-6h", "dt:>6h",
             "moved|dt:1-6h", "moved|dt:>6h"]


def format_table(name_to_summary, rows=None):
    rows = rows or ROW_ORDER
    names = list(name_to_summary)
    lines = []
    header = "%-24s" % "bucket" + "".join("%-30s" % n for n in names)
    lines.append(header)
    lines.append("-" * len(header))
    some = next(iter(name_to_summary.values()))
    for b in rows:
        if b not in some:
            continue
        cells = []
        for n in names:
            s = name_to_summary[n].get(b)
            cells.append("%-30s" % ("t1 %.3f rm %.3f nll %.2f sr %.2f" %
                         (s["top1"], s.get("top1_room", float("nan")),
                          s["nll"], s["search"]) if s else "-"))
        lines.append("%-24s" % ("%s(n=%d)" % (b, some[b]["n"])) + "".join(cells))
    return "\n".join(lines)
