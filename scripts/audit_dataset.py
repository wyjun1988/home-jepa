"""Dataset integrity audit. Verifies, per episode:

A. GT timeline well-formed (ordered, valid rooms, real changes)
B. Every POS observation matches GT (no hallucinated sightings)
C. SELF events match GT moves by the user (room + mover)
D. Query labels recomputable from raw data (gt_room, last_room/t, moved, dt, bin)
E. Query eligibility (not visible at qt, not OUT, seen before)
F. Causality of model windows (no event content from after qt except clamped
   spans; teacher window stops at the next sighting)
G. Anchor/room_feat consistency (the bug class we just hit)
H. Split disjointness + per-split moved stats
I. Regeneration determinism (same seed -> byte-identical episode)
"""
import argparse
import glob
import json
import os
import random
import sys
from bisect import bisect_right
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.model import EpTensors, ETYPES          # noqa: E402
from homejepa.sim import DTBINS, TICK_SEC             # noqa: E402

OUT = -1
FAIL = Counter()
WARN = Counter()
EXAMPLES = {}


def fail(key, example=None):
    FAIL[key] += 1
    if example is not None and key not in EXAMPLES:
        EXAMPLES[key] = example


def warn(key, example=None):
    WARN[key] += 1
    if example is not None and key not in EXAMPLES:
        EXAMPLES[key] = example


def dtbin(dt):
    for lo, hi, name in DTBINS:
        if lo <= dt < hi:
            return name
    return ">6h"


def gt_room_at(moves, t):
    """state(t) = after all moves strictly before t (same as sim/observation)."""
    i = bisect_right(moves, [t - 1, 10 ** 9]) - 1
    return moves[max(0, i)][1]


def audit_episode(ep, fp, deep_model_checks=True):
    rooms = set(r["id"] for r in ep["home"]["rooms"])
    locs = set(rc["id"] for rc in ep["home"]["recepts"])
    lroom = {rc["id"]: rc["room"] for rc in ep["home"]["recepts"]}
    gt = {g["obj"]: g["moves"] for g in ep["gt"]}
    T = ep["days"] * 17280

    # A. GT timeline
    for oid, ms in gt.items():
        for i, (t, room) in enumerate(ms):
            if room != OUT and room not in locs:
                fail("A.invalid_recept", (fp, oid, room))
            if i:
                if t <= ms[i - 1][0]:
                    fail("A.nonmonotonic_t", (fp, oid, t))
                if room == ms[i - 1][1]:
                    fail("A.no_change_move", (fp, oid, t))
            if not (0 <= t <= T + 17280):
                fail("A.time_range", (fp, oid, t))

    # B. positive sightings vs GT
    pos_runs = {}
    for e in ep["events"]:
        if e["type"] == "POS":
            oid, room = e["obj"], e["recept"]
            # endpoints are actual detections and must match GT; the interior
            # may bridge a <=3min departure+return (POS_GAP merge) -- that is
            # documented sensor-summary behavior, warn only
            for t in (e["t0"], e["t1"]):
                if gt_room_at(gt[oid], t) != room:
                    fail("B.pos_contradicts_gt", (fp, oid, t, room, gt_room_at(gt[oid], t)))
                    break
            else:
                mid = (e["t0"] + e["t1"]) // 2
                if gt_room_at(gt[oid], mid) != room:
                    warn("B.run_interior_bridged", (fp, oid, mid))
            pos_runs.setdefault(oid, []).append((e["t0"], e["t1"], room))
        if e["t1"] < e["t0"]:
            fail("B.run_inverted", (fp, e))
    for v in pos_runs.values():
        v.sort()

    # C. SELF events <-> user gt_moves, exact matching
    user_moves = {(m["obj"], m["t"]): m for m in ep["gt_moves"] if m["mover"] == 0}
    picks = {(e["obj"], e["t0"]): e for e in ep["events"] if e["type"] == "SELF_PICK"}
    drops = {(e["obj"], e["t0"]): e for e in ep["events"] if e["type"] == "SELF_DROP"}
    for key, m in user_moves.items():
        if m["src"] != OUT and key not in picks:
            fail("C.missing_selfpick", (fp, m))
        elif m["src"] != OUT and picks[key]["recept"] != m["src"]:
            fail("C.selfpick_room", (fp, m, picks[key]))
        if m["dst"] != OUT and key not in drops:
            fail("C.missing_selfdrop", (fp, m))
        elif m["dst"] != OUT and drops[key]["recept"] != m["dst"]:
            fail("C.selfdrop_room", (fp, m, drops[key]))
    for key, e in picks.items():
        # picks without a same-tick move are the carry-attach case ("picked it
        # up, moves later"); valid iff picked where the object actually was
        if key not in user_moves and gt_room_at(gt[e["obj"]], e["t0"]) != e["recept"]:
            fail("C.orphan_selfpick_wrong_room", (fp, e))
    for key in drops:
        if key not in user_moves:
            fail("C.orphan_selfdrop", (fp, drops[key]))

    # D/E. queries
    for q in ep["queries"]:
        oid, qt = q["obj"], q["qt"]
        runs = pos_runs.get(oid, [])
        i = bisect_right(runs, (qt, 10 ** 12, 10 ** 9)) - 1
        if i < 0:
            fail("E.no_prior_sighting", (fp, q))
            continue
        t0, t1, room = runs[i]
        if t0 <= qt <= t1 + 1:
            fail("E.visible_at_qt", (fp, q))
        if (room, t1) != (q["last_recept"], q["last_t"]):
            fail("D.last_seen_mismatch", (fp, q, (room, t1)))
        g = gt_room_at(gt[oid], qt)
        if g != q["gt_recept"]:
            fail("D.gt_room_mismatch", (fp, q, g))
        if g == OUT:
            fail("E.query_out", (fp, q))
        if q["moved"] != int(g != q["last_recept"]):
            fail("D.moved_flag", (fp, q))
        if q["moved_room"] != int(lroom[g] != lroom[q["last_recept"]]):
            fail("D.moved_room_flag", (fp, q))
        dt = (qt - t1) * TICK_SEC
        if dt != q["dt"] or dtbin(dt) != q["dtbin"]:
            fail("D.dt_mismatch", (fp, q, dt))
        movers = [m["mover"] for m in ep["gt_moves"] if m["obj"] == oid and t1 <= m["t"] < qt]
        if (movers[-1] if movers else None) != q["mover"]:
            fail("D.mover_mismatch", (fp, q, movers))
        if q["moved"] and not movers:
            fail("D.moved_without_move", (fp, q))

    if not deep_model_checks:
        return

    # F/G. model-input level (EpTensors)
    et = EpTensors(ep, max_events=256)
    know = (ETYPES["POS"], ETYPES["SELF_DROP"])
    for qi, q in enumerate(et.queries):
        qt = q["qt"]
        idxs = q["idxs"]
        if len(idxs) and int((et.t0[idxs] >= qt).sum()):
            fail("F.ctx_event_starts_after_qt", (fp, qi))
        # own-object guarantee: last know-event before qt is in the window
        last = None
        for j in et.obj_know.get(q["meta"]["obj"], []):
            if et.t1[j] < qt:
                last = j
        if last is not None and last not in set(idxs.tolist()):
            fail("F.own_last_event_truncated", (fp, qi))
        # anchor definition
        if last is not None:
            if q["anchor"] != et.loc_pos[int(et.ev_rec[last])]:
                fail("G.anchor_mismatch", (fp, qi))
        # teacher window: nothing after the next sighting
        nxt = None
        for j in et.obj_know.get(q["meta"]["obj"], []):
            if et.t1[j] >= qt:
                nxt = j
                break
        fut = q["fut"]
        if nxt is not None:
            if int((et.t1[fut] > et.t1[nxt]).sum()):
                fail("F.teacher_past_next_sighting", (fp, qi))
            if nxt not in set(fut.tolist()):
                fail("F.teacher_missing_reveal", (fp, qi))
        # room_feat vs anchor (the recent bug class)
        rf = q["loc_feat"]
        seen = [i for i in range(et.n_loc) if rf[i, 2] > 0]
        if seen:
            best = min(seen, key=lambda i: rf[i, 1])
            if best != q["anchor"] and abs(rf[best, 1] - rf[q["anchor"], 1]) > 1e-6:
                fail("G.loc_feat_anchor_mismatch", (fp, qi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "v0"))
    ap.add_argument("--train-sample", type=int, default=150)
    ap.add_argument("--regen-check", type=int, default=3)
    args = ap.parse_args()

    all_ids = {}
    n_done = 0
    for split in ["test", "val", "train"]:
        files = sorted(glob.glob(os.path.join(args.data, split, "ep_*.json")))
        all_ids[split] = set(os.path.basename(f) for f in files)
        use = files if split != "train" else files[:args.train_sample]
        moved = nq = 0
        for k, fp in enumerate(use):
            ep = json.load(open(fp))
            audit_episode(ep, os.path.basename(fp), deep_model_checks=(k % 3 == 0))
            for q in ep["queries"]:
                nq += 1
                moved += q["moved"]
            n_done += 1
        print("[%s] audited %d eps  queries %d  moved %.1f%%"
              % (split, len(use), nq, 100.0 * moved / max(1, nq)), flush=True)

    # H. split disjointness
    for a in all_ids:
        for b in all_ids:
            if a < b and all_ids[a] & all_ids[b]:
                fail("H.split_overlap", (a, b))

    # I. determinism: regenerate and byte-compare
    from homejepa.world import sample_home
    from homejepa.sim import simulate_episode
    files = sorted(glob.glob(os.path.join(args.data, "val", "ep_*.json")))[:args.regen_check]
    for fp in files:
        stored = json.load(open(fp))
        if "pack" in stored:      # pack-generated: regen needs the pack context
            continue
        seed = stored["seed"]
        rng = random.Random(seed)
        home = sample_home(rng, home_id=seed)
        fresh = simulate_episode(home, stored["days"], rng, len(stored["queries"]) or 120)
        fresh["seed"] = seed
        if json.dumps(fresh, sort_keys=True) != json.dumps(stored, sort_keys=True):
            fail("I.nondeterministic_regen", os.path.basename(fp))

    print("\n================ AUDIT RESULT (%d episodes) ================" % n_done)
    if not FAIL:
        print("PASS: no integrity failures")
    else:
        for k, v in sorted(FAIL.items()):
            print("FAIL %-32s x%d   e.g. %s" % (k, v, str(EXAMPLES.get(k))[:140]))
    for k, v in sorted(WARN.items()):
        print("warn %-32s x%d   e.g. %s" % (k, v, str(EXAMPLES.get(k))[:140]))


if __name__ == "__main__":
    main()
