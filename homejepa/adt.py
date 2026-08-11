"""ADT (Aria Digital Twin) -> v1 episodes.

Real-sensor validation set: mocap GT object poses + per-frame 2D visibility
(no vision model needed). Location = nearest static furniture surface
("receptacle"), rooms = k-means zones over furniture positions. One episode
per sequence; the wearer of that sequence's device is the observer, moves are
attributed to the housemate (ADT does not label movers). ~95 s sequences ->
short-horizon regime; tick = 5 s kept for feature consistency."""
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

TICK_NS = 5_000_000_000
FURN_CATS = {"couch": "sofa", "armchair": "sofa", "dining table": "dining_table",
             "table": "dining_table", "coffee table": "coffee_table",
             "console table": "console", "side table": "nightstand",
             "tv stand": "tv_stand", "bed frame": "bed", "dressing table": "desk",
             "cabinet": "cabinet_top", "cabinets and shelves": "cabinet_top",
             "shelf": "shelf", "bar stool": "shelf", "chair": "shelf",
             "dining chair": "shelf"}
CLS_MAP = {"food object": "snack_box", "tray": "snack_box", "container": "snack_box",
           "jar": "snack_box", "baking pan": "snack_box", "cutting board": "snack_box",
           "plate": "cup", "cup": "cup", "bowl": "cup",
           "bottle": "water_bottle", "can": "water_bottle",
           "spice grinder": "medicine", "book": "book", "box": "bag",
           "fork": "scissors", "spoon": "scissors", "kitchen utensil": "scissors",
           "coaster": "nail_clipper", "decorative accessory": "nail_clipper",
           "desk clock": "speaker", "play set": "book", "toy": "book",
           "vase": "plant"}
SIZE = {"snack_box": "m", "bag": "m", "plant": "m", "speaker": "m"}
RGB_STREAM = "214-1"
VIS_MIN = 0.15
HYST = 0.10        # m; switch nearest-receptacle only if closer by this margin
NEAR_MAX = 1.6     # m; farther than this from any receptacle -> floor


def _static_positions(seq_dir, ids):
    pos = {}
    with open(os.path.join(seq_dir, "scene_objects.csv")) as f:
        for row in csv.DictReader(f):
            uid = int(row["object_uid"])
            if uid in ids and int(row["timestamp[ns]"]) < 0:
                pos[uid] = (float(row["t_wo_x[m]"]), float(row["t_wo_y[m]"]),
                            float(row["t_wo_z[m]"]))
    return pos


def _dyn_tracks(seq_dir, ids):
    tr = defaultdict(list)
    with open(os.path.join(seq_dir, "scene_objects.csv")) as f:
        for row in csv.DictReader(f):
            uid = int(row["object_uid"])
            t = int(row["timestamp[ns]"])
            if uid in ids and t >= 0:
                tr[uid].append((t, float(row["t_wo_x[m]"]), float(row["t_wo_y[m]"]),
                                float(row["t_wo_z[m]"])))
    for v in tr.values():
        v.sort()
    return tr


def build_roster(gt_root, cap=28):
    """Global receptacle roster + room zones, shared by every sequence."""
    seq_dirs = sorted(d for d in glob.glob(os.path.join(gt_root, "*seq*"))
                      if os.path.isdir(d))
    inst = json.load(open(os.path.join(seq_dirs[0], "instances.json")))
    stat_ids = {int(k): v for k, v in inst.items()
                if v["motion_type"] == "static" and v["category"] in FURN_CATS}
    dyn_ids = {int(k): v for k, v in inst.items()
               if v["motion_type"] == "dynamic" and v["category"] in CLS_MAP}
    spos = _static_positions(seq_dirs[0], set(stat_ids))
    hosting = Counter()
    for sd in seq_dirs[::2]:
        tracks = _dyn_tracks(sd, set(dyn_ids))
        for uid, tr in tracks.items():
            for t, x, y, z in tr[:: max(1, len(tr) // 40)]:
                best, bd = None, NEAR_MAX
                for fid, (fx, fy, fz) in spos.items():
                    d = ((x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2) ** 0.5
                    if d < bd:
                        bd, best = d, fid
                if best is not None:
                    hosting[best] += 1
    keep = [fid for fid, _ in hosting.most_common(cap)]
    # rooms: k-means (k=4) over kept receptacle xz (y is up in ADT? use x,z)
    P = np.array([[spos[f][0], spos[f][2]] for f in keep])
    k = 4
    rng = np.random.RandomState(0)
    C = P[rng.choice(len(P), k, replace=False)]
    for _ in range(50):
        a = ((P[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([P[a == i].mean(0) if (a == i).any() else C[i] for i in range(k)])
    room_of_keep = {f: int(a[i]) for i, f in enumerate(keep)}
    # room types by content heuristic
    types = ["living_room"] * k
    for i in range(k):
        cats = [stat_ids[f]["category"] for f in keep if room_of_keep[f] == i]
        if any(c == "bed frame" for c in cats):
            types[i] = "bedroom"
        elif any(c == "dining table" for c in cats):
            types[i] = "dining_room"
    if types.count("living_room") > 1:
        types[types.index("living_room")] = "kitchen"
    return dict(seq_dirs=seq_dirs, stat_ids=stat_ids, dyn_ids=dyn_ids, spos=spos,
                keep=keep, room_of=room_of_keep, room_types=types, k=k)


def convert_sequence(seq_dir, roster, ep_id, queries_cap=60):
    inst_d = roster["dyn_ids"]
    keep, spos = roster["keep"], roster["spos"]
    k = roster["k"]
    # home dict (v1 schema): rooms + floor receptacle per room + kept furniture
    rooms = []
    tcount = Counter()
    for i in range(k):
        t = roster["room_types"][i]
        rooms.append(dict(id=i, type=t, tidx=tcount[t], recepts=[]))
        tcount[t] += 1
    recepts, loc_of_uid = [], {}
    rcount = Counter()
    for i in range(k):     # floor pseudo-receptacle per room
        recepts.append(dict(id=len(recepts), type="floor", tidx=rcount["floor"],
                            room=i, pos=[0.5, 0.5]))
        rcount["floor"] += 1
        rooms[i]["recepts"].append(recepts[-1]["id"])
    for f in keep:
        rt = FURN_CATS[roster["stat_ids"][f]["category"]]
        rid = roster["room_of"][f]
        recepts.append(dict(id=len(recepts), type=rt, tidx=rcount[rt], room=rid,
                            pos=[float(spos[f][0]), float(spos[f][2])]))
        rcount[rt] += 1
        rooms[rid]["recepts"].append(recepts[-1]["id"])
        loc_of_uid[f] = recepts[-1]["id"]
    floor_of_room = {i: rooms[i]["recepts"][0] for i in range(k)}

    tracks = _dyn_tracks(seq_dir, set(inst_d))
    t0 = min(tr[0][0] for tr in tracks.values() if tr)
    tmax = max(tr[-1][0] for tr in tracks.values() if tr)
    T = int((tmax - t0) / TICK_NS) + 2

    def tick(ns):
        return max(0, int((ns - t0) / TICK_NS))

    def nearest(x, y, z, prev_fid):
        best, bd = None, NEAR_MAX
        for fid in keep:
            fx, fy, fz = spos[fid]
            d = ((x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2) ** 0.5
            if d < bd:
                bd, best = d, fid
        if prev_fid is not None and best != prev_fid and prev_fid in keep:
            fx, fy, fz = spos[prev_fid]
            dp = ((x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2) ** 0.5
            if dp < bd + HYST:
                return prev_fid
        return best

    objects, oid_map = [], {}
    ccount = Counter()
    moves = {}
    gt_moves = []
    cls_of = {}
    for uid, v in sorted(inst_d.items()):
        tr = tracks.get(uid)
        if not tr:
            continue
        cls = CLS_MAP[v["category"]]
        oid = len(objects)
        oid_map[uid] = oid
        cls_of[oid] = cls
        prev_fid, cur_loc, ms = None, None, []
        # room of a floor fallback: room of nearest kept furniture even if far
        for j, (tns, x, y, z) in enumerate(tr):
            if j % 6:
                continue                     # ~5 Hz -> per-tick-ish sampling
            fid = nearest(x, y, z, prev_fid)
            if fid is None:
                dists = [(((x - spos[f][0]) ** 2 + (z - spos[f][2]) ** 2), f) for f in keep]
                loc = floor_of_room[roster["room_of"][min(dists)[1]]]
            else:
                loc = loc_of_uid[fid]
            prev_fid = fid
            tk = tick(tns)
            if cur_loc is None:
                ms.append((0, loc))
                cur_loc = loc
            elif loc != cur_loc:
                t_m = max(tk, ms[-1][0] + 1)
                ms.append((t_m, loc))
                gt_moves.append(dict(t=t_m, obj=oid, src=cur_loc, dst=loc,
                                     src_room=recepts[cur_loc]["room"],
                                     dst_room=recepts[loc]["room"],
                                     mover=1, reason="adt"))
                cur_loc = loc
        moves[oid] = ms
        objects.append(dict(id=oid, cls=cls, cidx=ccount[cls], owner=1,
                            size=SIZE.get(cls, "s"), home_recept=ms[0][1]))
        ccount[cls] += 1

    def loc_at(oid, tk):
        ms = moves[oid]
        lo = ms[0][1]          # state before any move = initial receptacle
        for t, loc in ms:
            if t <= tk - 1:
                lo = loc
            else:
                break
        return lo

    # observation events from real 2D visibility
    vis_dyn = defaultdict(set)      # tick -> {oid}
    vis_rec = defaultdict(set)      # tick -> {loc id}
    with open(os.path.join(seq_dir, "2d_bounding_box.csv")) as f:
        for row in csv.DictReader(f):
            if row["stream_id"] != RGB_STREAM:
                continue
            if float(row["visibility_ratio[%]"]) < VIS_MIN:
                continue
            uid = int(row["object_uid"])
            tk = tick(int(row["timestamp[ns]"]))
            if uid in oid_map:
                vis_dyn[tk].add(oid_map[uid])
            elif uid in loc_of_uid:
                vis_rec[tk].add(loc_of_uid[uid])

    events = []
    run = {}
    for tk in range(T):
        for oid in vis_dyn.get(tk, ()):
            loc = loc_at(oid, tk)
            r = run.get(oid)
            if r and r[2] == loc and tk - r[1] <= 6:
                r[1] = tk
            else:
                if r:
                    events.append(dict(t0=r[0], t1=r[1], type="POS", obj=oid,
                                       recept=r[2], room=recepts[r[2]]["room"]))
                run[oid] = [tk, tk, loc]
    for oid, r in run.items():
        events.append(dict(t0=r[0], t1=r[1], type="POS", obj=oid,
                           recept=r[2], room=recepts[r[2]]["room"]))
    nrun = None
    for tk in range(T):
        recs = vis_rec.get(tk)
        if not recs:
            continue
        room = Counter(recepts[g]["room"] for g in recs).most_common(1)[0][0]
        if nrun and nrun[0] == room and tk - nrun[2] <= 2:
            nrun[2] = tk
            nrun[3] |= recs
        else:
            if nrun:
                nrec = len(rooms[nrun[0]]["recepts"])
                events.append(dict(t0=nrun[1], t1=nrun[2], type="NEG", room=nrun[0],
                                   nz=nrec, cov=round(len(nrun[3]) / nrec, 3)))
            nrun = [room, tk, tk, set(recs)]
    if nrun:
        nrec = len(rooms[nrun[0]]["recepts"])
        events.append(dict(t0=nrun[1], t1=nrun[2], type="NEG", room=nrun[0],
                           nz=nrec, cov=round(len(nrun[3]) / nrec, 3)))
    events.sort(key=lambda e: (e["t1"], e["t0"], e["type"]))

    # glance sets from real co-visibility (3-tick = 15 s windows)
    glances = []
    GLW = 3
    for t0g in range(0, T, GLW):
        fset, oset, places = set(), set(), Counter()
        for tk in range(t0g, min(T, t0g + GLW)):
            for g in vis_rec.get(tk, ()):
                fset.add(g)
                places[g] += 1
            for oid in vis_dyn.get(tk, ()):
                oset.add(oid)
        furn = sorted(recepts[g]["type"] for g in fset)
        objs = sorted(cls_of[oid] for oid in oset)
        if furn or objs:
            glances.append(dict(t0=t0g, t1=min(T - 1, t0g + GLW - 1),
                                room=(recepts[places.most_common(1)[0][0]]["room"]
                                      if places else 0),
                                furn=sorted(furn), objs=sorted(objs),
                                gt_place=(places.most_common(1)[0][0] if places else -1)))

    pos_index = defaultdict(list)
    for e in events:
        if e["type"] == "POS":
            pos_index[e["obj"]].append((e["t0"], e["t1"], e["recept"]))
    for v in pos_index.values():
        v.sort()

    queries, seenq = [], set()
    def try_query(oid, qt):
        runs = pos_index.get(oid)
        if not runs or qt >= T:
            return
        import bisect
        i = bisect.bisect_right(runs, (qt, 10 ** 12, 10 ** 9)) - 1
        if i < 0:
            return
        rt0, rt1, rec = runs[i]
        if rt0 <= qt <= rt1 + 1:
            return
        dt = (qt - rt1) * 5
        if dt < 20:
            return
        key = (oid, qt // 3)
        if key in seenq:
            return
        seenq.add(key)
        g = loc_at(oid, qt)
        movers = [m["mover"] for m in gt_moves if m["obj"] == oid and rt1 <= m["t"] < qt]
        queries.append(dict(qt=qt, obj=oid, gt_recept=g, gt_room=recepts[g]["room"],
                            last_recept=rec, last_room=recepts[rec]["room"], last_t=rt1,
                            dt=dt, moved=int(g != rec),
                            moved_room=int(recepts[g]["room"] != recepts[rec]["room"]),
                            mover=(movers[-1] if movers else None),
                            n_moves=len(movers),
                            dtbin="<10m" if dt < 600 else "10-60m"))
    for oid in moves:
        for qt in range(8, T, 4):
            try_query(oid, qt)
    for m in gt_moves:
        for off in (4, 6, 9, 13, 18):
            try_query(m["obj"], m["t"] + off)
    queries = queries[:queries_cap]

    return dict(
        home=dict(id=ep_id, n_agents=2, rooms=rooms, recepts=recepts,
                  objects=objects),
        days=1, events=events, glances=glances,
        gt=[dict(obj=oid, moves=[list(x) for x in ms]) for oid, ms in sorted(moves.items())],
        gt_moves=gt_moves, queries=queries, seed=ep_id)
