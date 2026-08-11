"""World sampling: rooms/zones/objects/agents. State space is room-level;
zones exist only for the observation model (DESIGN_20260807.md §1-2)."""
import math

TICKS_PER_MIN = 12          # tick = 5s
TICKS_PER_DAY = 1440 * TICKS_PER_MIN
OUT = -1                    # virtual location: object/agent outside the home

ROOM_TYPES = ["living_room", "kitchen", "bedroom", "bathroom",
              "study", "entrance", "balcony", "dining_room"]

SIZE_PDET = {"s": 0.75, "m": 0.90, "l": 0.98}

# v1: receptacles (surfaces) per room type. Every room also gets a "floor".
# First entries are more likely to be sampled; count 2-5 per room.
RECEPT_TYPES = ["floor", "sofa", "coffee_table", "tv_stand", "shelf",
                "counter", "sink", "dining_table", "fridge_top", "cabinet_top",
                "bed", "nightstand", "wardrobe_top", "desk", "drawer_top",
                "bath_counter", "shoe_cabinet", "console", "sideboard"]
RECEPT_IDX = {r: i for i, r in enumerate(RECEPT_TYPES)}
ROOM_RECEPTS = {
    "living_room": ["sofa", "coffee_table", "tv_stand", "shelf"],
    "kitchen": ["counter", "sink", "dining_table", "fridge_top", "cabinet_top"],
    "bedroom": ["bed", "nightstand", "desk", "wardrobe_top"],
    "bathroom": ["bath_counter", "shelf"],
    "study": ["desk", "shelf", "drawer_top"],
    "entrance": ["shoe_cabinet", "console"],
    "balcony": ["shelf"],
    "dining_room": ["dining_table", "sideboard", "shelf"],
}

# home: room-type weights for the object's canonical location
# per_agent: one instance per agent (owned); count: instances per home otherwise
OBJECT_CLASSES = {
    "phone":        dict(size="s", home={"bedroom": .4, "living_room": .4, "study": .2}, per_agent=True),
    "keys":         dict(size="s", home={"entrance": .7, "living_room": .3}, per_agent=True),
    "wallet":       dict(size="s", home={"entrance": .4, "bedroom": .4, "living_room": .2}, per_agent=True),
    "earbuds":      dict(size="s", home={"bedroom": .4, "living_room": .3, "study": .3}, per_agent=True),
    "bag":          dict(size="m", home={"bedroom": .5, "entrance": .3, "living_room": .2}, per_agent=True),
    "watch":        dict(size="s", home={"bedroom": .7, "living_room": .3}, count=(0, 2)),
    "glasses":      dict(size="s", home={"bedroom": .5, "study": .3, "living_room": .2}, count=(0, 2)),
    "laptop":       dict(size="m", home={"study": .5, "living_room": .3, "bedroom": .2}, count=(1, 3)),
    "charger":      dict(size="s", home={"bedroom": .4, "study": .4, "living_room": .2}, count=(1, 3)),
    "tablet":       dict(size="m", home={"living_room": .5, "bedroom": .5}, count=(0, 2)),
    "book":         dict(size="s", home={"study": .4, "bedroom": .3, "living_room": .3}, count=(1, 4)),
    "notebook":     dict(size="s", home={"study": .6, "living_room": .4}, count=(0, 2)),
    "pen":          dict(size="s", home={"study": .6, "living_room": .4}, count=(0, 3)),
    "cup":          dict(size="s", home={"kitchen": .6, "living_room": .2, "study": .2}, count=(2, 4)),
    "water_bottle": dict(size="s", home={"kitchen": .5, "bedroom": .3, "living_room": .2}, count=(0, 3)),
    "remote":       dict(size="s", home={"living_room": 1.0}, count=(1, 2)),
    "snack_box":    dict(size="m", home={"kitchen": .8, "living_room": .2}, count=(0, 2)),
    "scissors":     dict(size="s", home={"kitchen": .5, "study": .5}, count=(0, 2)),
    "nail_clipper": dict(size="s", home={"living_room": .5, "bedroom": .5}, count=(0, 1)),
    "medicine":     dict(size="s", home={"kitchen": .4, "living_room": .3, "bathroom": .3}, count=(0, 2)),
    "umbrella":     dict(size="m", home={"entrance": 1.0}, count=(0, 2)),
    "speaker":      dict(size="m", home={"living_room": .7, "bedroom": .3}, count=(0, 1)),
    "plant":        dict(size="l", home={"living_room": .5, "balcony": .5}, count=(0, 2)),
    "hair_dryer":   dict(size="m", home={"bathroom": .7, "bedroom": .3}, count=(0, 1)),
}
CLASS_NAMES = sorted(OBJECT_CLASSES.keys())
CLASS_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# class -> preferred home receptacle types (fallback: any receptacle in room)
HOME_RECEPT = {
    "phone": ["nightstand", "desk", "sofa"], "keys": ["console", "shoe_cabinet", "counter"],
    "wallet": ["console", "desk", "nightstand"], "earbuds": ["desk", "nightstand", "coffee_table"],
    "bag": ["floor", "console", "wardrobe_top"], "watch": ["nightstand", "desk"],
    "glasses": ["nightstand", "desk"], "laptop": ["desk", "coffee_table"],
    "charger": ["nightstand", "desk"], "tablet": ["sofa", "coffee_table", "bed"],
    "book": ["shelf", "desk", "nightstand"], "notebook": ["desk", "shelf"], "pen": ["desk"],
    "cup": ["counter", "cabinet_top", "sink"], "water_bottle": ["counter", "desk"],
    "remote": ["coffee_table", "sofa", "tv_stand"], "snack_box": ["cabinet_top", "counter"],
    "scissors": ["drawer_top", "counter"], "nail_clipper": ["drawer_top", "nightstand"],
    "medicine": ["cabinet_top", "bath_counter"], "umbrella": ["shoe_cabinet", "floor"],
    "speaker": ["shelf", "tv_stand"], "plant": ["shelf", "floor"],
    "hair_dryer": ["bath_counter", "wardrobe_top"],
}


def _sample_zones(rng, n):
    """n zone centers in the unit square, min separation 0.3."""
    pts = []
    for _ in range(200):
        if len(pts) == n:
            break
        p = (rng.uniform(0.12, 0.88), rng.uniform(0.12, 0.88))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= 0.09 for q in pts):
            pts.append(p)
    while len(pts) < n:
        pts.append((rng.uniform(0.12, 0.88), rng.uniform(0.12, 0.88)))
    return [[round(x, 3), round(y, 3)] for x, y in pts]


def sample_persona(rng, is_user):
    out_style = rng.random()
    return dict(
        wake_m=int(rng.gauss(7.2 * 60, 45)),
        sleep_m=int(rng.gauss(23.3 * 60, 50)),
        tidiness=rng.uniform(0.15, 0.85),
        # 0: commuter (long daily out), 1: part-time out, 2: mostly home
        out_style=0 if out_style < 0.45 else (1 if out_style < 0.75 else 2),
        out_leave_m=int(rng.gauss(8.6 * 60, 40)),
        out_ret_m=int(rng.gauss(18.6 * 60, 70)),
        act_pref={a: rng.uniform(0.5, 1.5) for a in
                  ["work", "tv", "read", "tablet_time", "phone_scroll", "chores", "snack", "coffee", "idle"]},
        tidy_daily=rng.uniform(0.35, 1.0),  # prob of a tidy-up pass per day
        is_user=is_user,
    )


def sample_home(rng, home_id, n_agents=None):
    if n_agents is None:
        n_agents = 1 + rng.randint(1, 3)          # user + 1..3 housemates
    n_bedrooms = min(3, max(1, n_agents - (1 if (n_agents > 2 and rng.random() < 0.4) else 0)))

    types = ["living_room", "kitchen", "bathroom"] + ["bedroom"] * n_bedrooms
    for extra, p in [("entrance", 0.7), ("study", 0.5), ("balcony", 0.4), ("dining_room", 0.3)]:
        if len(types) >= 8:
            break
        if rng.random() < p:
            types.append(extra)
    rooms, seen = [], {}
    recepts = []
    for t in types:
        k = seen.get(t, 0)
        seen[t] = k + 1
        rid = len(rooms)
        prefs = ROOM_RECEPTS.get(t, ["shelf"])
        n_extra = min(len(prefs), rng.randint(1, 4))
        rtypes = ["floor"] + prefs[:n_extra]
        pos = _sample_zones(rng, len(rtypes))
        rlist = []
        for rt, p in zip(rtypes, pos):
            rlist.append(len(recepts))
            recepts.append(dict(id=len(recepts), type=rt, room=rid, pos=p))
        rooms.append(dict(id=rid, type=t, tidx=k, recepts=rlist))
    rcount = {}
    for rc in recepts:
        rc["tidx"] = rcount.get(rc["type"], 0)
        rcount[rc["type"]] = rc["tidx"] + 1
    rooms_by_type = {}
    for r in rooms:
        rooms_by_type.setdefault(r["type"], []).append(r["id"])

    agents = [sample_persona(rng, i == 0) for i in range(n_agents)]
    beds = rooms_by_type["bedroom"]
    for i, a in enumerate(agents):
        a["bedroom"] = beds[i % len(beds)]

    def pick_home_room(cls):
        w = [(rid, OBJECT_CLASSES[cls]["home"].get(rooms[rid]["type"], 0.0) /
              max(1, len(rooms_by_type.get(rooms[rid]["type"], []))))
             for rid in range(len(rooms))]
        tot = sum(x for _, x in w)
        if tot <= 0:
            return rng.randrange(len(rooms))
        u = rng.random() * tot
        for rid, x in w:
            u -= x
            if u <= 0:
                return rid
        return w[-1][0]

    def pick_recept(cls, rid):
        cands = rooms[rid]["recepts"]
        prefs = HOME_RECEPT.get(cls, [])
        for pt in prefs:
            hit = [g for g in cands if recepts[g]["type"] == pt]
            if hit:
                return hit[0]
        return rng.choice(cands)

    objects = []
    for cls in CLASS_NAMES:
        spec = OBJECT_CLASSES[cls]
        if spec.get("per_agent"):
            n, owners = n_agents, list(range(n_agents))
        else:
            lo, hi = spec["count"]
            n, owners = rng.randint(lo, hi), None
        for k in range(n):
            hr = pick_home_room(cls)
            hrec = pick_recept(cls, hr)
            irec = hrec
            if rng.random() < 0.15:                      # start misplaced
                irec = rng.choice(rooms[rng.randrange(len(rooms))]["recepts"])
            objects.append(dict(
                id=len(objects), cls=cls, cidx=k,
                owner=(owners[k] if owners else None),
                size=spec["size"], home_recept=hrec, init_recept=irec))
    return dict(id=home_id, rooms=rooms, recepts=recepts,
                rooms_by_type=rooms_by_type,
                objects=objects, agents=agents, n_agents=n_agents)
