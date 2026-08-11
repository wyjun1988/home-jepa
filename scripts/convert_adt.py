import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.adt import build_roster, convert_sequence   # noqa: E402


def main():
    root = os.path.join(os.path.dirname(__file__), "..", "data", "adt", "gt")
    out = os.path.join(os.path.dirname(__file__), "..", "data", "adt", "eps", "test")
    os.makedirs(out, exist_ok=True)
    roster = build_roster(root)
    print("roster: %d receptacles, %d dynamic objects, rooms=%s"
          % (len(roster["keep"]), len(roster["dyn_ids"]), roster["room_types"]), flush=True)
    n = nq = nmoved = nmr = 0
    for i, sd in enumerate(roster["seq_dirs"]):
        ep = convert_sequence(sd, roster, ep_id=800000 + i)
        with open(os.path.join(out, "ep_%06d.json" % (800000 + i)), "w") as f:
            json.dump(ep, f)
        n += 1
        nq += len(ep["queries"])
        nmoved += sum(q["moved"] for q in ep["queries"])
        nmr += sum(q["moved_room"] for q in ep["queries"])
    print("episodes %d  queries %d  moved %.1f%%  moved_room %.1f%%"
          % (n, nq, 100.0 * nmoved / max(1, nq), 100.0 * nmr / max(1, nq)))


if __name__ == "__main__":
    main()
