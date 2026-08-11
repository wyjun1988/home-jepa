"""Routine schedules: per agent per day, a list of segments
(t0, t1, activity, room). Times in ticks. Object interactions are resolved
later in sim.py from each segment's `uses` template."""
from .world import TICKS_PER_MIN, TICKS_PER_DAY, OUT

# rooms: preference order; "bedroom.own" resolves to the agent's bedroom.
# recepts: preferred surface types in the activity room (fallback: any).
# dur in minutes. uses: class -> fetch probability.
ACTIVITIES = {
    "bathroom_morning": dict(rooms=["bathroom"], recepts=["bath_counter"], dur=(8, 20),
                             uses={"phone": .25}),
    "meal":             dict(rooms=["dining_room", "kitchen"], recepts=["dining_table", "counter"],
                             dur=(20, 45), uses={"phone": .5, "cup": .5}),
    "coffee":           dict(rooms=["kitchen"], recepts=["counter", "dining_table"], dur=(5, 12),
                             uses={"cup": .7, "phone": .35}),
    "work":             dict(rooms=["study", "living_room", "bedroom.own"],
                             recepts=["desk", "coffee_table", "dining_table"], dur=(60, 180),
                             uses={"laptop": .85, "charger": .35, "earbuds": .3, "notebook": .25,
                                   "pen": .25, "cup": .45, "phone": .55}),
    "tv":               dict(rooms=["living_room"], recepts=["coffee_table", "sofa"], dur=(40, 150),
                             uses={"remote": .9, "phone": .55, "snack_box": .3, "cup": .3}),
    "read":             dict(rooms=["bedroom.own", "living_room", "study"],
                             recepts=["bed", "sofa", "desk"], dur=(30, 90),
                             uses={"book": .85, "glasses": .35}),
    "tablet_time":      dict(rooms=["living_room", "bedroom.own"], recepts=["sofa", "bed"],
                             dur=(30, 90), uses={"tablet": .7, "earbuds": .3}),
    "phone_scroll":     dict(rooms=["bedroom.own", "living_room"], recepts=["bed", "sofa"],
                             dur=(15, 50), uses={"phone": .95, "charger": .25}),
    "chores":           dict(rooms=["balcony", "bathroom", "kitchen"], recepts=["counter", "shelf"],
                             dur=(15, 40), uses={"phone": .2}),
    "shower":           dict(rooms=["bathroom"], recepts=["bath_counter"], dur=(12, 25),
                             uses={"hair_dryer": .3}),
    "groom":            dict(rooms=["bathroom", "bedroom.own"], recepts=["bath_counter", "nightstand"],
                             dur=(8, 15), uses={"hair_dryer": .25, "watch": .2, "glasses": .2}),
    "snack":            dict(rooms=["kitchen", "living_room"], recepts=["counter", "coffee_table"],
                             dur=(8, 15), uses={"snack_box": .5, "cup": .4, "phone": .4}),
    "cook":             dict(rooms=["kitchen"], recepts=["counter"], dur=(25, 60),
                             uses={"snack_box": .8, "scissors": .45, "cup": .3,
                                   "water_bottle": .35, "phone": .3}),
    "wash_dishes":      dict(rooms=["kitchen"], recepts=["sink"], dur=(8, 20), uses={}),
    "idle":             dict(rooms=["living_room", "bedroom.own"], recepts=["sofa", "bed"],
                             dur=(20, 60), uses={}),
}
# after use: same-room shift destinations (dishes to the sink etc.)
WITHIN_SHIFT = {"cup": ["sink", "counter"], "snack_box": ["cabinet_top", "counter"],
                "water_bottle": ["counter", "sink"]}

# v1.1 furniture-lifecycle chains (HOMER+ transfer gap fix): after use, some
# classes follow a typed destination distribution instead of the generic
# home/stay split -- the dish cycle (use -> sink -> cupboard) and the food
# cycle (fridge/cabinet -> counter/table -> back) that real routines showed.
# entries: (dest, prob); dest = receptacle TYPE, or "home"/"stay".
CLASS_CHAIN = {
    "cup":          [("sink", .40), ("counter", .10), ("home", .20), ("stay", .30)],
    "water_bottle": [("sink", .15), ("fridge_top", .25), ("home", .20), ("stay", .40)],
    "snack_box":    [("fridge_top", .25), ("cabinet_top", .25), ("home", .15), ("stay", .35)],
    "scissors":     [("drawer_top", .30), ("home", .25), ("stay", .45)],
    "medicine":     [("cabinet_top", .35), ("home", .25), ("stay", .40)],
    "hair_dryer":   [("bath_counter", .30), ("wardrobe_top", .20), ("home", .20), ("stay", .30)],
}
DISH_CLASSES = ("cup",)     # wash_dishes moves these from sink -> cabinet/home
GO_OUT_TAKES = {"keys": .9, "wallet": .75, "phone": .95, "bag": .45, "umbrella": .08}
BLOCK_ACTS = ["work", "tv", "read", "tablet_time", "phone_scroll", "chores", "snack", "coffee", "idle"]


def resolve_room(pref_list, agent, home, rng):
    cands = []
    for p in pref_list:
        if p == "bedroom.own":
            cands.append(agent["bedroom"])
        else:
            cands.extend(home["rooms_by_type"].get(p, []))
    if not cands:
        # fall back to any common room the home actually has (HOMER+ homes
        # have no living_room, e.g.)
        for t in ("living_room", "dining_room", "study", "bedroom"):
            if t in home["rooms_by_type"]:
                return home["rooms_by_type"][t][0]
        return home["rooms"][0]["id"]
    return cands[0] if rng.random() < 0.7 else rng.choice(cands)


def _dur_ticks(act, rng, table=None):
    lo, hi = (table or ACTIVITIES)[act]["dur"]
    return rng.randint(lo, hi) * TICKS_PER_MIN


def build_agent_day(agent, aidx, day, home, rng):
    """Return list of segments dict(t0,t1,act,room,agent). room==OUT while out;
    act in {'sleep','out'} are special."""
    base = day * TICKS_PER_DAY
    acts = home.get("pack_acts", ACTIVITIES)
    timed = timed_for_day(home, aidx, day, rng)
    segs = []
    t = base + max(5 * 60, min(10 * 60, int(rng.gauss(agent["wake_m"], 25)))) * TICKS_PER_MIN

    def add(act, room=None, dur=None):
        nonlocal t
        d = dur if dur is not None else _dur_ticks(act, rng, acts)
        r = room if room is not None else resolve_room(acts[act]["rooms"], agent, home, rng)
        segs.append(dict(t0=t, t1=t + d, act=act, room=r, agent=aidx))
        t += d

    def flush_timed():
        while timed and t >= timed[0][0]:
            add(timed.pop(0)[1])

    add("bathroom_morning")
    if rng.random() < 0.5:
        add("groom")
    add("meal")
    flush_timed()

    # main out block
    style = agent["out_style"]
    big_out = (style == 0 and rng.random() < 0.9) or (style == 1 and rng.random() < 0.45)
    if big_out:
        leave = base + int(rng.gauss(agent["out_leave_m"], 25)) * TICKS_PER_MIN
        ret = base + int(rng.gauss(agent["out_ret_m"], 45)) * TICKS_PER_MIN
        leave = max(leave, t)
        ret = max(ret, leave + 60 * TICKS_PER_MIN)
        if leave > t:
            add("idle", dur=leave - t)
        segs.append(dict(t0=t, t1=ret, act="out", room=OUT, agent=aidx))
        t = ret
        flush_timed()
        if rng.random() < 0.4:
            add("shower")
    errand_done = False

    sleep_t = base + max(21 * 60, min(26 * 60, int(rng.gauss(agent["sleep_m"], 30)))) * TICKS_PER_MIN
    dinner_t = base + int(rng.gauss(19.0 * 60, 30)) * TICKS_PER_MIN
    had_dinner = False
    prefs = agent["act_pref"]
    while t < sleep_t - 20 * TICKS_PER_MIN:
        flush_timed()
        if not had_dinner and t >= dinner_t:
            if rng.random() < 0.6 and "kitchen" in home["rooms_by_type"]:
                add("cook")
            add("meal")
            if rng.random() < 0.55 and "kitchen" in home["rooms_by_type"]:
                add("wash_dishes")
            had_dinner = True
            continue
        if (not big_out) and (not errand_done) and rng.random() < 0.12:
            d = rng.randint(30, 120) * TICKS_PER_MIN
            segs.append(dict(t0=t, t1=t + d, act="out", room=OUT, agent=aidx))
            t += d
            errand_done = True
            continue
        cand = [a for a in BLOCK_ACTS]
        w = [prefs.get(a, 1.0) for a in cand]
        u = rng.random() * sum(w)
        for a, x in zip(cand, w):
            u -= x
            if u <= 0:
                act = a
                break
        else:
            act = "idle"
        remain = (sleep_t - 10 * TICKS_PER_MIN) - t
        d = min(_dur_ticks(act, rng), max(10 * TICKS_PER_MIN, remain))
        add(act, dur=d)
    if rng.random() < 0.6:
        add("phone_scroll", room=agent["bedroom"], dur=rng.randint(10, 30) * TICKS_PER_MIN)
    # sleep till next day's start (capped at day end; next day starts with its own wake)
    segs.append(dict(t0=t, t1=(day + 1) * TICKS_PER_DAY, act="sleep", room=agent["bedroom"], agent=aidx))
    return segs


def build_schedules(home, days, rng):
    per_agent = []
    for aidx, agent in enumerate(home["agents"]):
        segs = []
        for d in range(days):
            segs.extend(build_agent_day(agent, aidx, d, home, rng))
        # clip overlaps caused by day overflow: keep chronological, drop inverted
        segs = [s for s in segs if s["t1"] > s["t0"]]
        per_agent.append(segs)
    return per_agent


# ---------------- LLM-authored household packs (behavior layer) ----------------
# A pack supplies personas, signature activities (with times, used objects and
# after-use destination chains) and chain overrides. The simulator remains the
# physics/observation layer; the audit still gates everything. See
# docs/DESIGN §10 (LLM pack pipeline).

def apply_pack(home, pack, rng):
    """Merge a pack into a sampled home: persona fields, per-home activity
    table (base + signature), and per-home class chains."""
    acts = dict(ACTIVITIES)
    for sa in pack.get("activities", []):
        acts[sa["name"]] = dict(
            rooms=sa.get("rooms", ["living_room"]),
            recepts=sa.get("recepts", []),
            dur=tuple(sa.get("dur", (10, 30))),
            uses=sa.get("uses", {}),
            after=sa.get("after", {}))
    home["pack_acts"] = acts
    home["pack_chain"] = {**CLASS_CHAIN,
                          **{k: [tuple(x) for x in v]
                             for k, v in pack.get("chains", {}).items()}}
    home["pack_timed"] = pack.get("activities", [])
    for aidx, agent in enumerate(home["agents"]):
        pp = pack.get("personas", [])
        if aidx < len(pp):
            p = pp[aidx]
            agent["wake_m"] = int(p.get("wake_h", agent["wake_m"] / 60) * 60)
            agent["sleep_m"] = int(p.get("sleep_h", agent["sleep_m"] / 60) * 60)
            agent["tidiness"] = float(p.get("tidiness", agent["tidiness"]))
            agent["out_style"] = int(p.get("out_style", agent["out_style"]))
            agent["out_leave_m"] = int(p.get("out_leave_h", agent["out_leave_m"] / 60) * 60)
            agent["out_ret_m"] = int(p.get("out_ret_h", agent["out_ret_m"] / 60) * 60)
            for a, w in p.get("act_pref", {}).items():
                agent["act_pref"][a] = float(w)
    return home


def timed_for_day(home, aidx, day, rng):
    """Pack signature activities scheduled for this agent on this day, as a
    list of (start_tick, name) sorted by time."""
    out = []
    base = day * TICKS_PER_DAY
    for sa in home.get("pack_timed", []):
        if sa.get("agent", 0) != aidx:
            continue
        if rng.random() >= sa.get("p_day", 1.0):
            continue
        h = sa.get("time_h")
        if h is None:
            continue
        jit = sa.get("jitter_m", 25)
        t = base + int((h * 60 + rng.gauss(0, jit)) * TICKS_PER_MIN / 1)
        out.append((max(base, t), sa["name"]))
    return sorted(out)
