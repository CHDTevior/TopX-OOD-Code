"""Per-joint text descriptions built from the skeleton's geometry for joints whose names carry no anatomy."""
import re
from collections import Counter, defaultdict

import numpy as np


SIDE_TOL = 0.02


PLACE_TOL = 0.02


ANCHOR_MAX_HOPS = 4


LOW_INFO = re.compile(
    r"^(?:(?:Left|Right|Upper|Lower|Front|Back|Hind|Middle)\s+)*"
    r"(?:Bone|Root|Object|Joint|Node|Armature|Dummy|Mesh|Null|Empty|Group|Unnamed|Unknown)"
    r"(?:\s+End)?$", re.I)


PLAN_WORD = {"bipedal": "bipedal", "quadrupedal": "quadrupedal", "insectoid": "insectoid",
             "avian": "avian", "marine": "marine", "serpentine": "serpentine",
             "articulated_rigid": "rigid articulated", "uncertain": None}


NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
       "eleven", "twelve"]


ORD = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
       "ninth", "tenth", "eleventh", "twelfth"]


def num(n: int) -> str:
    return NUM[n] if n < len(NUM) else str(n)


def ordn(n: int) -> str:
    return ORD[n] if n < len(ORD) else f"{n}th"


def is_low_info(d: str) -> bool:
    s = d[:-len(" joint.")] if d.endswith(" joint.") else d
    return bool(LOW_INFO.match(s))


def plan_phrase(plan: str | None) -> str:
    w = PLAN_WORD.get(plan or "", None)
    if w is None:
        return "a skeleton"
    return ("an " if w[0] in "aeiou" else "a ") + w + " skeleton"


def describe_rig(names, desc, parents, P, s_rig, plan):
    """Return the rig's description list with every placeholder row replaced."""
    J = len(names)
    low = [is_low_info(d) for d in desc]
    children = defaultdict(list)
    for j in range(J):
        p = int(parents[j])
        if p >= 0:
            children[p].append(j)
    depth = np.zeros(J, np.int64)
    for j in range(1, J):
        depth[j] = depth[int(parents[j])] + 1

    x = P[:, 0] / s_rig
    y = (P[:, 1] - P[0, 1]) / s_rig          # root-relative height
    z = P[:, 2] / s_rig                      # already root-centred in z

    def side_of(j):
        if x[j] > SIDE_TOL:
            return "left"
        if x[j] < -SIDE_TOL:
            return "right"
        return "mid-line"

    # ---- maximal unbranched chains -------------------------------------------------------
    heads = [j for j in range(1, J) if int(parents[j]) == 0 or len(children[int(parents[j])]) >= 2]
    chain_of, chain_pos, chain_len, chain_end = {}, {}, {}, {}
    for h in heads:
        cur, seq = h, [h]
        while len(children[cur]) == 1:
            cur = children[cur][0]
            seq.append(cur)
        for i, j in enumerate(seq):
            chain_of[j] = h
            chain_pos[j] = i + 1
            chain_len[j] = len(seq)
            chain_end[j] = seq[-1]

    # ---- sibling limb ranking: same parent, same side, front (max z) first ---------------
    limb_rank, limb_group = {}, {}
    for p, ch in children.items():
        hs = [c for c in ch if c in chain_of and chain_of[c] == c]
        by_side = defaultdict(list)
        for c in hs:
            by_side[side_of(c)].append(c)
        for sd, group in by_side.items():
            group.sort(key=lambda c: (-z[c], -y[c], c))
            for i, c in enumerate(group):
                limb_rank[c], limb_group[c] = i + 1, len(group)

    # ---- nearest non-placeholder ancestor -------------------------------------------------
    stat = {"anchored": 0, "side_dropped": 0, "anchor_dropped": 0, "above": 0, "below": 0,
            "level": 0}

    def anchor(j):
        """(name, hops, ancestor index) of the nearest named ancestor, or (None, 0, -1).

        The name loses a LEADING left/right word that the ancestor's own rest x refutes: the anchor
        is quoted verbatim out of the corpus, and an inherited side word the geometry contradicts
        would put a claim in the sentence that nothing supports.  A side word on a mid-line ancestor
        is left alone -- the deadband means "not measurable here", not "wrong", and SIDE_TOL
        mis-files 1.48% of genuinely lateral joints as central.
        """
        hops, a = 1, int(parents[j])
        while a >= 0 and low[a] and hops <= ANCHOR_MAX_HOPS:
            a, hops = int(parents[a]), hops + 1
        if not (a >= 0 and not low[a] and hops <= ANCHOR_MAX_HOPS):
            return None, 0, -1
        stem = desc[a][:-len(" joint.")] if desc[a].endswith(" joint.") else desc[a]
        toks = stem.strip().lower().split()
        if toks and toks[0] in ("left", "right") and side_of(a) not in (toks[0], "mid-line"):
            stat["side_dropped"] += 1
            toks = toks[1:]
        if not toks:                       # the name was NOTHING but a refuted side word
            stat["anchor_dropped"] += 1
            return None, 0, -1
        return " ".join(toks), hops, a

    def place_clause(j):
        v = []
        if y[j] > PLACE_TOL:
            v.append("above")
        elif y[j] < -PLACE_TOL:
            v.append("below")
        h = None
        if z[j] > PLACE_TOL:
            h = "in front of"
        elif z[j] < -PLACE_TOL:
            h = "behind"
        if v and h:
            return f"{v[0]} and {h} the root"
        if v:
            return f"{v[0]} the root"
        if h:
            return f"{h} the root"
        return "level with the root"

    out = list(desc)
    for j in range(J):
        if not low[j]:
            continue
        pl = plan_phrase(plan)
        if j == 0:
            out[j] = f"the root joint at the base of {pl}"
            continue
        h = chain_of[j]
        L, pos, end = chain_len[j], chain_pos[j], chain_end[j]
        nkids = len(children[end])
        # "limb" ends in a tip; "branching chain" runs on into further limbs.
        kind = "limb" if nkids == 0 else "branching chain"
        if L == 1:
            role = "a single-bone limb" if nkids == 0 else "a single joint"
        elif pos == L and nkids == 0:
            role = f"the tip of a {num(L)}-bone limb"
        else:
            role = f"the {ordn(pos)} joint of a {num(L)}-bone {kind}"
        sd = side_of(j)
        parts = [f"{role} " + ("on the mid-line of " if sd == "mid-line"
                               else f"on the {sd} side of ") + pl]
        if limb_group.get(h, 1) > 1:
            parts.append(f"{ordn(limb_rank[h])} limb from the front")
        a_name, a_hops, a_idx = anchor(j)
        if a_name:
            # ancestry = hops along the tree; vertical wording only where the height says so
            hop = "one bone" if a_hops == 1 else f"{num(a_hops)} bones"
            dy = y[j] - y[a_idx]                       # measured, in rig scales
            rel = (" and above it" if dy > PLACE_TOL else
                   " and below it" if dy < -PLACE_TOL else "")
            stat["above" if dy > PLACE_TOL else "below" if dy < -PLACE_TOL else "level"] += 1
            stat["anchored"] += 1
            parts.append(f"{hop} down the chain from the {a_name}{rel}")
        else:
            parts.append(place_clause(j))
        if pos == L and nkids >= 2:
            parts.append(f"where {num(nkids)} limbs branch")
        out[j] = ", ".join(parts)
    return out, low, stat
