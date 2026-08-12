"""Is the headline number even interpretable? (three checks)

Q1  room-level rollup vs receptacle-level: how much of "moved" difficulty is
    within-room surface churn, which a user standing in the room would just see?
Q2  user-moved queries: the drop location is physically known, so accuracy
    should be ~1.0. Measure how often the (noid) anchor actually equals the
    true location, i.e. whether the hand-chain recovered the information.
Q3  housemate-moved queries split by EVIDENCE AVAILABLE in the log before qt:
      none      : old surface never re-scanned, object never seen elsewhere
                  -> rationally unanswerable; last-seen IS the right answer
      negative  : old surface was well-seen without it -> knows it left
      positive  : same class seen at another surface   -> can associate
    A single averaged number over these three is not meaningful.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from homejepa.model import load_split, make_batch      # noqa: E402
from reeval import build_supervised, build_jepa_probe  # noqa: E402


def evidence_class(ep, q):
    """Classify what the log offers about this query, before qt (id-free)."""
    qt = q["qt"]
    meta = q["meta"]
    anchor = q["anchor"]                      # slot index
    cls_i = q["qcls"]
    saw_elsewhere = False
    neg_at_anchor = False
    for g in ep.glances:
        if g["t1"] >= qt:
            break
        objs = g["objs"]
        furn = set(g["furn"])
        anchor_type = ep.recepts[anchor]["type"]
        from homejepa.world import CLASS_NAMES
        cname = CLASS_NAMES[cls_i]
        if anchor_type in furn and len(objs) >= 2 and cname not in objs:
            neg_at_anchor = True
    # positive evidence: a POS event of the same class at a different surface,
    # after the anchor time
    same_inst = False
    for j in ep.cls_ev.get(cls_i, []):
        if ep.t1[j] >= qt or ep.t1[j] <= q["anchor_t"]:
            continue
        if ep.et[j] == 0 and ep.ev_rec[j] >= 0 and ep.loc_pos[int(ep.ev_rec[j])] != anchor:
            saw_elsewhere = True
            if int(ep.ev_obj[j]) == meta["obj"]:
                same_inst = True
                break
    if saw_elsewhere:
        # is that sighting actually the queried instance (true evidence) or a
        # different instance of the same class (a distractor)? GT used for
        # DIAGNOSIS only -- never reaches the model.
        return "positive-true(진짜 그 물건)" if same_inst else "positive-false(동클래스 방해물)"
    if neg_at_anchor:
        return "negative(원위치 부재 확인)"
    return "none(단서 없음)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "v4", "test"))
    ap.add_argument("--ckpts", default="supervised_two_head_v4no2,jepa_jepa_v4no2_ft_probe,supervised_two_head_v4id2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()
    dev = torch.device(args.device)
    files = sorted(glob.glob(os.path.join(args.data, "ep_*.json")))

    cache = {}
    for name in args.ckpts.split(","):
        ck = torch.load(os.path.join(os.path.dirname(__file__), "..", "results", name + ".pt"),
                        map_location=dev)
        noid = bool(ck.get("args", {}).get("noid", False))
        if noid not in cache:
            cache[noid] = load_split(files, 256, noid=noid)
        eps = cache[noid]
        model, _ = (build_jepa_probe if name.startswith("jepa_") else build_supervised)(ck, dev)
        model = model.to(dev).eval()

        samples = [(e, q) for e in range(len(eps)) for q in range(len(eps[e].queries))]
        hit1 = Counter(); hit2 = Counter(); tot = Counter()
        room1 = Counter()
        anchor_correct = Counter(); wouldbe = Counter()
        with torch.no_grad():
            for i in range(0, len(samples), 256):
                chunk = samples[i:i + 256]
                b = make_batch(eps, chunk, dev)
                p = model.log_prob(b).exp().cpu()
                for j, (ei, qi) in enumerate(chunk):
                    ep = eps[ei]
                    q = ep.queries[qi]
                    m = q["meta"]
                    gt = q["gt"]
                    pv = p[j, :ep.n_loc]
                    order = pv.argsort(descending=True)
                    # room rollup
                    rp = defaultdict(float)
                    for k in range(ep.n_loc):
                        rp[ep.loc_room[ep.recepts[k]["id"]]] += float(pv[k])
                    gtroom = ep.loc_room[ep.recepts[gt]["id"]]
                    keys = []
                    if m["moved"]:
                        keys.append("moved_room" if m["moved_room"] else "moved_within")
                        if m.get("mover") == 0:
                            keys.append("user_moved")
                        elif m.get("mover") is not None:
                            keys.append("housemate:" + evidence_class(ep, q))
                    else:
                        keys.append("stayed")
                    # would-be anchor if noid used class-level POS *or SELF_DROP*
                    wb = q["anchor"]
                    best_t = q["anchor_t"]
                    for jj in ep.cls_ev.get(q["qcls"], []):
                        if ep.t1[jj] < q["qt"] and ep.t1[jj] > best_t \
                                and ep.et[jj] == 3 and ep.ev_rec[jj] >= 0:
                            best_t = int(ep.t1[jj])
                            wb = ep.loc_pos[int(ep.ev_rec[jj])]
                    for k in keys:
                        tot[k] += 1
                        wouldbe[k] += int(wb == gt)
                        hit1[k] += int(order[0].item() == gt)
                        hit2[k] += int(gt in order[:2].tolist())
                        room1[k] += int(max(rp, key=rp.get) == gtroom)
                        anchor_correct[k] += int(q["anchor"] == gt)
        print("\n=== %s (%s) ===" % (name, "noid" if noid else "id"))
        print("%-34s %6s %7s %7s %8s %9s %9s" % ("버킷", "n", "top1", "top2", "방top1", "앵커=정답", "수정앵커"))
        for k in sorted(tot, key=lambda x: -tot[x]):
            n = tot[k]
            print("%-34s %6d %7.3f %7.3f %8.3f %9.3f %9.3f"
                  % (k, n, hit1[k] / n, hit2[k] / n, room1[k] / n,
                     anchor_correct[k] / n, wouldbe[k] / n))
        del model


if __name__ == "__main__":
    main()
