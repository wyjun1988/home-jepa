"""Diagnostic: how much of the observation history actually reaches the model?
Checks whether the query object's last sighting (POS or SELF_DROP) survives the
max_events truncation window."""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.model import EpTensors, ETYPES   # noqa: E402

TICK_SEC = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "v0", "test"))
    ap.add_argument("--max-events", type=int, default=256)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "ep_*.json")))[:args.limit]
    n_ev, span_h, cover, cover_moved, cover_stayed = [], [], [], [], []
    cover_by_dt = {}
    cover_user_moved = []
    for fp in files:
        raw = json.load(open(fp))
        ep = EpTensors(raw, args.max_events)
        n_ev.append(len(ep.t1))
        objid = [e.get("obj") for e in raw["events"]]
        etype = [e["type"] for e in raw["events"]]
        eroom = [e.get("room") for e in raw["events"]]
        for q in ep.queries:
            idxs = q["idxs"]
            qt = q["qt"]
            m = q["meta"]
            if len(idxs):
                span_h.append((qt - ep.t0[idxs[0]]) * TICK_SEC / 3600.0)
            # last evidence about THIS object anywhere before qt
            last_ev = None
            for j in range(len(objid)):
                if objid[j] == m["obj"] and etype[j] in ("POS", "SELF_DROP") and ep.t1[j] < qt:
                    last_ev = j
            hit = int(last_ev is not None and last_ev in set(idxs.tolist()))
            cover.append(hit)
            (cover_moved if m["moved"] else cover_stayed).append(hit)
            cover_by_dt.setdefault(m["dtbin"], []).append(hit)
            if m["moved"] and m.get("mover") == 0:
                cover_user_moved.append(hit)

    def pct(v):
        return 100.0 * sum(v) / max(1, len(v))

    print("max_events = %d" % args.max_events)
    print("events/episode: mean %.0f  max %d" % (np.mean(n_ev), max(n_ev)))
    print("window time span: median %.1f h  p10 %.1f h  (episode = 72 h)"
          % (np.median(span_h), np.percentile(span_h, 10)))
    print("\nquery object's last sighting inside the window:")
    print("  overall      %.1f%%  (n=%d)" % (pct(cover), len(cover)))
    print("  stayed       %.1f%%" % pct(cover_stayed))
    print("  moved        %.1f%%" % pct(cover_moved))
    print("  moved_by:user%.1f%%" % pct(cover_user_moved))
    for b in ["<10m", "10-60m", "1-6h", ">6h"]:
        if b in cover_by_dt:
            print("  dt %-7s  %.1f%%  (n=%d)" % (b, pct(cover_by_dt[b]), len(cover_by_dt[b])))


if __name__ == "__main__":
    main()
