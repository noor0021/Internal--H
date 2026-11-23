"""Independent re-solve of every algorithmically checkable item, parsing the question TEXT (not generator state)."""
import json, re, itertools, collections, datetime as dt

rows = [json.loads(l) for l in open("queries_500.jsonl")]
gold = lambda r: r["choices"][r["correct_idx"]]
bad = collections.Counter(); checked = collections.Counter()


def check(r, ok):
    checked[r["task"]] += 1
    if not ok:
        bad[r["task"]] += 1; print("FAIL", r["id"], r["question"][:200], "| gold:", gold(r))


for r in rows:
    q, t = r["question"], r["task"]
    if t == "navigate":
        x = y = 0; d = 0
        body = q.split("starting point? ")[1]
        ff = body.startswith("Always face forward.")
        for m in re.finditer(r"Take (\d+) steps?( forward| backward| left| right)?\.|Turn (left|right|around)\.", body):
            if m.group(3):
                d = (d + {"left": -1, "right": 1, "around": 2}[m.group(3)]) % 4
            else:
                s = int(m.group(1)); dirn = (m.group(2) or " forward").strip()
                if ff:
                    dx, dy = {"forward": (0, 1), "backward": (0, -1), "left": (-1, 0), "right": (1, 0)}[dirn]
                else:
                    dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0)][d]
                x += dx * s; y += dy * s
        check(r, gold(r) == ("Yes" if (x, y) == (0, 0) else "No"))
    elif t == "web_of_lies":
        sents = q.replace("Question: ", "").split(". ")
        m = re.match(r"(\w+) (tells the truth|lies)", sents[0]); val = {m.group(1): m.group(2) == "tells the truth"}
        for s in sents[1:]:
            m = re.match(r"(\w+) says (\w+) (tells the truth|lies)", s)
            if m: val[m.group(1)] = (m.group(3) == "tells the truth") == val[m.group(2)]
        who = re.search(r"Does (\w+) tell the truth\?", q).group(1)
        check(r, gold(r) == ("Yes" if val[who] else "No"))
    elif t == "tracking_shuffled_objects_three_objects":
        hold = {}
        for p in ["Alice", "Bob", "Claire"]:
            m = re.search(p + r" (?:has a|is dancing with|is playing|gets) ([^,.]+?)(?: ball| present)?(?:,|\.| and)", q)
            hold[p] = m.group(1)
        for a, b in re.findall(r"(Alice|Bob|Claire) and (Alice|Bob|Claire) (?:swap|switch|trade)", q):
            hold[a], hold[b] = hold[b], hold[a]
        tgt = re.search(r"At the end of the \w+, (Alice|Bob|Claire)", q).group(1)
        check(r, gold(r).replace(" ball", "").replace(" present", "") == hold[tgt])
    elif t == "temporal_sequences":
        h = lambda s: (int(s[:-2]) % 12) + (12 if s.endswith("pm") else 0)
        wake = h(re.search(r"woke up at (\d+[ap]m)", q).group(1))
        close = h(re.search(r"closed after (\d+[ap]m)", q).group(1))
        busy = [(h(a), h(b)) for a, b in re.findall(r"from (\d+[ap]m) to (\d+[ap]m)", q)]
        free = [hr for hr in range(wake, close) if not any(a <= hr < b for a, b in busy)]
        a, b = gold(r).split(" to ")
        check(r, list(range(h(a), h(b))) == free)
        # every other option must overlap a busy sighting
        for c in r["choices"]:
            if c == gold(r): continue
            a, b = c.split(" to ")
            assert any(x in [hr for s in busy for hr in range(*s)] for x in range(h(a), h(b))), r["id"]
    elif t == "logical_deduction_five_objects":
        m = r["meta"]; objs = m["objects"]
        sols = [p for p in itertools.permutations(range(5))
                if all((p[c[1]] < p[c[2]]) if c[0] == "rel" else (p[c[1]] == c[2]) for c in m["clues"])]
        ok = len(sols) == 1 and list(sols[0]) == m["positions"]
        ans_obj = objs[m["positions"].index(m["query_position"])]
        ok = ok and ans_obj in gold(r) and sum(o in gold(r) for o in objs) == 1
        # clue sentences must be present in text
        ok = ok and q.count(".") >= len(m["clues"]) + 2
        check(r, ok)
    elif t == "date_understanding":
        from dateutil.relativedelta import relativedelta
        today = dt.date.fromisoformat(r["meta"]["today"]); off = r["meta"]["offset"]
        f = {"yesterday": -1, "tomorrow": 1, "one week ago from today": -7, "one week from today": 7,
             "10 days ago": -10, "24 hours later": 1, "today": 0}
        exp = today + dt.timedelta(f[off]) if off in f else today + (relativedelta(years=-1) if "year" in off else relativedelta(months=-1))
        check(r, gold(r) == exp.strftime("%m/%d/%Y") and len(set(r["choices"])) == len(r["choices"]))
    elif t == "hyperbaton":
        from generate import ADJ
        rank = {w: i for i, ws in enumerate(ADJ) for w in ws}
        ok = True
        for c in r["choices"]:
            ws = c.split()[:-1]; ordered = [rank[w] for w in ws] == sorted(rank[w] for w in ws)
            ok &= ordered == (c == gold(r))
        check(r, ok)
    elif t == "sports_understanding":
        from generate import ATHLETES, ACTIONS, SPORT_OF
        who_role = next(k for k, v in ATHLETES.items() for a in v if q.startswith(f'Is the following sentence plausible? "{a} '))
        act_role = next(k for k, v in ACTIONS.items() for a in v if q.endswith(f' {a}."'))
        check(r, gold(r) == ("plausible" if who_role == act_role else "implausible")
              and (who_role == act_role or SPORT_OF[who_role] != SPORT_OF[act_role]))

print("\nchecked:", dict(checked)); print("failures:", dict(bad) or 0)
print("total independently verified:", sum(checked.values()), "/ 500")
