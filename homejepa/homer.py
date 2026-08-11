"""HOMER+ -> v1 episodes at FURNITURE (receptacle) level.

The HOMER+ resident is the housemate (crowdsourced-real routines); the
glasses-wearing observer is synthesized with our persona/FOV machinery.
v1 upgrade: their Furniture/Appliance nodes map 1:1 to receptacles, so the
~60 furniture-level moves per day (vs 3-5 room-level) become the signal."""
import random

from .sim import _Sim, TICKS_PER_MIN
from .world import OUT, sample_persona, _sample_zones
from .routines import ACTIVITIES, build_schedules

ROOM_MAP = {"bathroom": "bathroom", "dining_room": "dining_room",
            "bedroom": "bedroom", "home_office": "study",
            "kitchen": "kitchen", "livingroom": "living_room",
            "living_room": "living_room"}
FURN_MAP = {"bathroom_cabinet": "cabinet_top", "bathroom_counter": "bath_counter",
            "bookshelf": "shelf", "chair": "shelf", "coffe_maker": "counter",
            "cupboard": "cabinet_top", "desk": "desk", "dresser": "wardrobe_top",
            "filing_cabinet": "drawer_top", "fridge": "fridge_top",
            "kitchen_cabinet": "cabinet_top", "kitchen_counter": "counter",
            "microwave": "counter", "sink": "sink", "sofa": "sofa",
            "stove": "counter", "table": "dining_table", "tvstand": "tv_stand"}
CLS_MAP = {
    "book": "book", "notebook": "notebook", "pen": "pen", "mail": "notebook",
    "cd": "book", "chessboard": "book", "deck_of_cards": "book",
    "cd_player": "speaker", "radio": "speaker", "instrument_guitar": "speaker",
    "vacuum_cleaner": "speaker", "headset": "earbuds",
    "remote_control": "remote", "hairdryer": "hair_dryer",
    "painkillers": "medicine", "tooth_paste": "medicine", "shaving_cream": "medicine",
    "hairbrush": "nail_clipper", "razor": "nail_clipper", "toothbrush": "nail_clipper",
    "washcloth": "nail_clipper",
    "cup": "cup", "mug": "cup", "coffee_cup": "cup", "drinking_glass": "cup",
    "wine_glass": "cup", "plate": "cup", "bowl": "cup", "toothbrush_holder": "cup",
    "juice": "water_bottle", "milk": "water_bottle", "oil": "water_bottle",
    "wine": "water_bottle", "cleaning_solution": "water_bottle",
    "fork": "scissors", "spoon": "scissors", "knife": "scissors",
    "clothes_jacket": "bag", "groceries": "bag", "trashbag": "bag",
    "coffee": "snack_box", "coffee_filter": "snack_box", "ground_coffee": "snack_box",
    "cookingpot": "snack_box", "cutting_board": "snack_box", "fryingpan": "snack_box",
    "dry_pasta": "snack_box", "tea": "snack_box",
    "food_apple": "snack_box", "food_bread": "snack_box", "food_cereal": "snack_box",
    "food_cheese": "snack_box", "food_chicken": "snack_box", "food_donut": "snack_box",
    "food_jam": "snack_box", "food_oatmeal": "snack_box",
    "food_peanut_butter": "snack_box", "food_rice": "snack_box",
    "food_vegetable": "snack_box",
}
SIZE = {"speaker": "m", "bag": "m", "snack_box": "m", "hair_dryer": "m"}


def _graph_maps(g):
    rooms = {n["id"]: n["class_name"] for n in g["nodes"] if n.get("category") == "Rooms"}
    furn = {n["id"]: n["class_name"] for n in g["nodes"]
            if n.get("category") in ("Furniture", "Appliances") and n["class_name"] in FURN_MAP}
    parent = {}
    for e in g["edges"]:
        if e["relation_type"] in ("INSIDE", "ON"):
            parent.setdefault(e["from_id"], e["to_id"])
    return rooms, furn, parent


def _object_locs(g, rooms, furn, parent):
    """Each placable object -> (furniture id or None, room id)."""
    out = {}
    for n in g["nodes"]:
        if n.get("category") != "placable_objects":
            continue
        cur, seen, hit_f = n["id"], set(), None
        while cur is not None and cur not in rooms and cur not in seen:
            seen.add(cur)
            if cur != n["id"] and cur in furn and hit_f is None:
                hit_f = cur
            cur = parent.get(cur)
        if cur in rooms:
            out[n["id"]] = (hit_f, cur)
    return out


def convert_day(day, seed, ep_id):
    rng = random.Random(seed)
    g0 = day["graphs"][0]
    hrooms, hfurn, parent0 = _graph_maps(g0)

    rooms, seen_t, rid_map = [], {}, {}
    for hid, cname in sorted(hrooms.items()):
        t = ROOM_MAP.get(cname, "living_room")
        k = seen_t.get(t, 0)
        seen_t[t] = k + 1
        rid_map[hid] = len(rooms)
        rooms.append(dict(id=len(rooms), type=t, tidx=k, recepts=[]))
    rooms_by_type = {}
    for r in rooms:
        rooms_by_type.setdefault(r["type"], []).append(r["id"])

    # receptacles: floor per room + their furniture (positions synthesized for FOV)
    recepts, rec_map, rcount = [], {}, {}
    for r in rooms:
        recepts.append(dict(id=len(recepts), type="floor", tidx=rcount.get("floor", 0),
                            room=r["id"], pos=[0.5, 0.5]))
        rcount["floor"] = rcount.get("floor", 0) + 1
        r["recepts"].append(recepts[-1]["id"])
    furn_room = {}
    for hid, cname in sorted(hfurn.items()):
        cur, seen = hid, set()
        while cur is not None and cur not in hrooms and cur not in seen:
            seen.add(cur)
            cur = parent0.get(cur)
        furn_room[hid] = rid_map[cur] if cur in hrooms else 0
    for rid in range(len(rooms)):
        mine = [h for h, rr in furn_room.items() if rr == rid]
        pos = _sample_zones(rng, max(1, len(mine)))
        for h, p in zip(sorted(mine), pos):
            rt = FURN_MAP[hfurn[h]]
            rec_map[h] = len(recepts)
            recepts.append(dict(id=len(recepts), type=rt, tidx=rcount.get(rt, 0),
                                room=rid, pos=p))
            rcount[rt] = rcount.get(rt, 0) + 1
            rooms[rid]["recepts"].append(recepts[-1]["id"])
    floor_of = {r["id"]: r["recepts"][0] for r in rooms}

    def loc_of(entry):
        f, hroom = entry
        return rec_map.get(f, floor_of[rid_map[hroom]])

    objects, oid_map, cls_count = [], {}, {}
    init = _object_locs(g0, hrooms, hfurn, parent0)
    for n in sorted(g0["nodes"], key=lambda x: x["id"]):
        if n.get("category") != "placable_objects" or n["id"] not in init:
            continue
        cls = CLS_MAP.get(n["class_name"], "snack_box")
        k = cls_count.get(cls, 0)
        cls_count[cls] = k + 1
        oid_map[n["id"]] = len(objects)
        objects.append(dict(id=len(objects), cls=cls, cidx=k, owner=1,
                            size=SIZE.get(cls, "s"),
                            home_recept=loc_of(init[n["id"]]),
                            init_recept=loc_of(init[n["id"]])))

    agents = [sample_persona(rng, True), sample_persona(rng, False)]
    for a in agents:
        a["bedroom"] = rooms_by_type.get("bedroom", [rooms[0]["id"]])[0]
    home = dict(id=ep_id, rooms=rooms, recepts=recepts, rooms_by_type=rooms_by_type,
                objects=objects, agents=agents, n_agents=2)

    sim = _Sim(home, days=1, rng=rng, queries_per_ep=80)
    cur = {hid: loc_of(v) for hid, v in init.items()}
    for gi in range(1, len(day["graphs"])):
        t = max(1, int(round(day["times"][gi] * TICKS_PER_MIN)))
        rooms_g, furn_g, parent_g = _graph_maps(day["graphs"][gi])
        now = _object_locs(day["graphs"][gi], rooms_g, furn_g, parent_g)
        for hid, entry in now.items():
            if hid not in oid_map:
                continue
            loc = loc_of(entry)
            if cur.get(hid) is not None and loc != cur[hid]:
                sim._move(oid_map[hid], t, loc, mover=1,
                          reason="homer:" + str(day["activities"][gi]))
            cur[hid] = loc

    schedules = build_schedules(home, 1, rng)
    for s in schedules[0]:
        if s["room"] != OUT:
            s["recept"] = sim.room_recept(s["room"],
                                          ACTIVITIES.get(s["act"], {}).get("recepts"))
    sim.run_observation(schedules)
    queries = sim.sample_queries()
    seen = {(q["obj"], q["qt"] // (30 * TICKS_PER_MIN)) for q in queries}
    for m in sim.gt_moves:
        for off_min in (30, 120, 360):
            qt = m["t"] + off_min * TICKS_PER_MIN + rng.randint(0, 20 * TICKS_PER_MIN)
            if qt >= sim.T or not sim.awake[qt]:
                continue
            q = sim.make_query(m["obj"], qt)
            if q is None or q["dt"] < 300:
                continue
            key = (q["obj"], q["qt"] // (30 * TICKS_PER_MIN))
            if key in seen:
                continue
            seen.add(key)
            queries.append(q)
    return dict(
        home=dict(id=home["id"], n_agents=2,
                  rooms=[dict(id=r["id"], type=r["type"], tidx=r["tidx"],
                              recepts=r["recepts"]) for r in rooms],
                  recepts=recepts,
                  objects=[dict(id=o["id"], cls=o["cls"], cidx=o["cidx"], owner=o["owner"],
                                size=o["size"], home_recept=o["home_recept"])
                           for o in objects]),
        days=1, events=sim.events, glances=sim.glances,
        gt=[dict(obj=oid, moves=ms) for oid, ms in sorted(sim.moves.items())],
        gt_moves=sim.gt_moves, queries=queries, seed=seed)
