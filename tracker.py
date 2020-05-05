import argparse
import queue

import levedata


class ExperiencePoints:
    levels = {
        1: 0,
        2: 300,
        3: 900,
        4: 2000,
        5: 3700,
        6: 6000,
        7: 10200,
        8: 16200,
        9: 23550,
        10:	33480,
        11:	45280,
        12:	60880,
        13:	80480,
        14:	104180,
        15:	130580,
        16:	161080,
        17:	196480,
        18:	236980,
        19:	282680,
        20:	333680,
        21:	390280,
        22:	454180,
        23:	525580,
        24:	604680,
        25:	691780,
        26:	786980,
        27:	896780,
        28:	1021580,
        29:	1161780,
        30:	1317680,
        31:	1480180,
        32:	1656080,
        33:	1845680,
        34:	2049180,
        35:	2267080,
        36:	2499400,
        37:	2749300,
        38:	3017100,
        39:	3303300,
        40:	3608200,
        41:	3932200,
        42:	4272400,
        43:	4629200,
        44:	5002900,
        45:	5393700,
        46:	5801900,
        47:	6239500,
        48:	6707000,
        49:	7205000,
        50:	7734000,
        51:	8598000,
        52:	9656400,
        53:	10923600,
        54:	12478800,
        55:	14350800,
        56:	16568400,
        57:	19160400,
        58:	22155600,
        59:	25582800,
        60:	29470800,
        61:	33940800,
        62:	38813800,
        63:	44129800,
        64:	49938800,
        65:	56302800,
        66:	63297800,
        67:	71019800,
        68:	79594800,
        69:	89187800,
        70:	100013800,
        71:	112462800,
        72:	126343800,
        73:	141899800,
        74:	159398400,
        75:	179148400,
        76:	201478400,
        77:	226818400,
        78:	255468400,
        79:	288218400,
        80:	325868400,
    }
    @staticmethod
    def get_level(exp:int) -> int:
        """ Get the current level based on total exp

            Params:
                exp: A non-negative number representing a character's total
                     exp

            Returns:
                :int: Character's current level given the total `exp`

            Raises ValueError when exp is negative
        """
        if exp < 0:
            raise ValueError("`exp` cannot be negative")
        return max(filter(lambda x: x[1] <= exp,
                         ExperiencePoints.levels.items()))[0]

    @staticmethod
    def get_total_exp(level:int, exp:int=0) -> int:
        """ Gets a character's total exp given level and current exp

            Params:
                :level: Character's current level. Must be a valid level
                :exp: character's current exp, if any

            Returns:
                :int: Character's total exp

            Raises ValueError if level is not within [1..80]
        """
        try:
            cur_exp = ExperiencePoints.levels[level]
        except KeyError:
            raise ValueError(f"`{level}` is not a valid level [1..80]")
        return cur_exp + exp


class Step:
    def __init__(self, leve, level, exp, next_exp, repeats=None):
        self.leve = leve
        self.level = level
        self.exp = exp
        self.next_exp = next_exp
        if repeats is None:
            self.repeats = leve.repeats + 1
        else:
            self.repeats = repeats

    def hash(self):
        return (self.leve, self.repeats)

    @staticmethod
    def format(count, step):
        repeats = step.repeats + (step.leve.repeats+1)*(count-1)
        max_repeats = count * (step.leve.repeats + 1)
        return (f"{count}x {step.leve} (Repeats: {repeats}/{max_repeats}) "
                f"Lvl: {step.level}[{step.exp:,}/{step.next_exp:,}]\n")

    # def __str__(self):
    #     return (f"{self.leve} (Repeats: {self.repeats}/{self.leve.repeats+1}) "
    #             f"(Lvl: {self.level}[{self.exp:,}/{self.next_exp:,}])")


class LeveExpTracker(object):
    tiers = list(range(5, 50, 5)) + list(range(50, 80, 2))
    # Leves below this cutoff give 3000xp to characters at or above this level
    hard_cutoff = 70
    def __init__(self, job, level, cur_exp=0, target=80, max_steps=100):
        self.cur_exp = ExperiencePoints.get_total_exp(level, cur_exp)
        self.job = job
        self.target = target
        self.weights = dict(allowances=1,
                        travel=0,
                        items=0)
        self.max_steps = max_steps
        self.steps = list()
        self.leves = list(filter(lambda x: x.job == self.job, levedata.leves))
        self.debug = False
        self.force_leve_order = True
        self.last_leve = None
        self._level = None
        self.max_allowances = 1
        self.partial_repeats = False
        self.min_repeats = 0
        self.restrict_leves = True
        self.never_restrict_levels = False

    @property
    def level(self):
        if self._level is None:
            self._level = ExperiencePoints.get_level(self.cur_exp)
        return self._level

    def add_step(self, leve, repeats=None):
        if repeats is None:
            num_repeats = leve.repeats + 1
        else:
            num_repeats = min(repeats, leve.repeats + 1)

        for _ in range(0, num_repeats):
            if self.level >= self.hard_cutoff and leve.level < self.hard_cutoff:
                # After the hard cutoff, any previous levequests reward 3000xp
                exp = 3000
            else:
                exp = leve.exp
            self.cur_exp += exp * 2
            self._level = None  # Bust the level cache

        cur_exp = self.cur_exp - ExperiencePoints.levels[self.level]
        if self.level+1 in ExperiencePoints.levels:
            next_exp = ExperiencePoints.levels[self.level+1] - ExperiencePoints.levels[self.level]
        else:
            next_exp = 0
        step = Step(leve, self.level, cur_exp, next_exp, num_repeats)
        self.steps.append(step)
        if repeats == leve.repeats + 1:
            self.last_leve = leve
        else:
            self.last_leve = None

    def __repr__(self):
        return str(self)

    def __str__(self):
        out = f"\nCost: {self.cost}\nLevel: {self.level}\nSteps:\n"
        last = None
        count = 0
        for step in self.steps:
            if last is None or step.leve is not last.leve:
                if count > 0:
                    out += Step.format(count, last)
                count = 1
                last = step
            else:
                count += 1
                last = step
        out += Step.format(count, last) + "\n"
        return out

    @property
    def available_leves(self):
        if self.restrict_leves:
            if self.level >= 70:
                min_level = 70
            else:
                min_level = max(self.level - 5, 0)
        else:
            min_level = 1
            self.restrict_leves = not self.never_restrict_levels

        if self.last_leve is not None:
            # Always include the most recent leve
            yield self.last_leve

        leves = set(filter(lambda leve: leve.job == self.job and \
                                        min_level <= leve.level <= self.level and \
                                        leve.allowances <= self.max_allowances and \
                                        leve.repeats >= self.min_repeats,
                           self.leves))
        leves -= set(map(lambda x: x.leve, self.steps))
        sorted_leves = sorted(
            list(leves),
            key=lambda leve: leve.exp * (leve.repeats+1) / leve.allowances)
        yield from sorted_leves

    def clone(self):
        ret = LeveExpTracker(self.job, self.level, 0, self.target)
        ret.cur_exp = self.cur_exp
        ret.steps = list(self.steps)
        return ret

    @property
    def cost(self):
        return sum(
            map(lambda step: step.leve.allowances * self.weights['allowances'] + \
                             step.leve.count * (step.repeats) * self.weights['items'],
                self.steps)
        )

    @property
    def preference(self):
        return (
            sum([step.leve.preferred for step in self.steps]),
            -len(set(map(lambda x: x.leve, self.steps))),
            -sum(map(lambda x: x.repeats, self.steps))
        )

    def run(self):
        working = queue.Queue()
        working.put(self)
        clones = list()
        depth = 0
        costs = {self.level: float("inf")}
        iter_count = 0
        self.restrict_leves = False

        while working.qsize() > 0:
            node = working.get()
            best = None
            if clones:
                best = clones[0].cost
            print(f"\rQueue: {working.qsize()} - Level: {max(costs.keys())} - "
                  f"Depth: {depth} - Found: {len(clones)} - "
                  f"Iter: {iter_count} - Best Cost: {best} - costs: "
                  f"{','.join(map(str, costs.values()))}",
                  end='',
                  flush=True)
            iter_count += 1
            if iter_count % 1000 == 0:
                pass
                # print(clone)

            if len(node.steps) > depth:
                depth = len(node.steps)

            for leve in node.available_leves:
                if self.partial_repeats:
                    min_repeats = 1
                    max_repeats = leve.repeats + 1
                else:
                    min_repeats = leve.repeats + 1
                    max_repeats = leve.repeats + 1

                for num_repeats in range(min_repeats, max_repeats + 1):
                    clone = node.clone()
                    pre_level = clone.level
                    clone.add_step(leve, repeats=num_repeats)
                    for completed_level in range(pre_level, clone.level):
                        if completed_level in costs:
                            if clone.cost < costs[completed_level]:
                                # print(f"\nFound new best: {clone}")
                                costs[completed_level] = clone.cost
                        else:
                            # print(f"\nFound new level: {clone}")
                            costs[completed_level] = clone.cost

                    if clone.level in costs and \
                            clone.cost >= costs[clone.level]:
                        if self.debug:
                            print(f"\nFailed at same-level check: "
                                  f"{clone.cost}/{clone.level} vs {costs}")
                        continue

                    # Check for target completion
                    if clone.level >= clone.target:
                        if len(clones) == 0:
                            clones.append(clone)
                        elif clone.cost < clones[0].cost:
                            clones = [clone]
                        elif clone.cost == clones[0].cost:
                            clones.append(clone)
                        continue

                    if self.debug:
                        print(f"Still going... {clone.cost} {clone.level}")

                    working.put(clone)

        best = max([clone.preference for clone in clones])
        return list(filter(lambda clone: clone.preference == best, clones))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("LeveKit Calculator")
    parser.add_argument("job", help="The three-letter capitalized job acronym (e.g.: ALC)")
    parser.add_argument("level", type=int, help="The current level")
    parser.add_argument("target", type=int, help="The goal level", default=80)
    parser.add_argument("--exp", "-e", type=int, help="Current exp in current level", default=0)
    parser.add_argument("--verbose", "-v", action="store_true", help="LOUD NOISES")
    parser.add_argument("--temple", "-t", action="store_true", help="Consider Temple Leves. They're usually awful, and I haven't found an optimal purpose for them yet, but hey! Maybe!")
    parser.add_argument("--partial-repeats", "-p", action="store_true", help="Consider ending repeatable quests without turning in three times. Much slower, but might find better options")
    parser.add_argument("--repeat-only", "-r", action="store_true", help="Consider only repeatable quests (if available). Much faster, but potentially inaccurate")
    args = parser.parse_args()
    tracker = LeveExpTracker(args.job, args.level, args.exp, args.target)
    if args.temple:
        tracker.max_allowances = 10

    tracker.partial_repeats = args.partial_repeats

    if args.repeat_only:
        tracker.min_repeats = 2

    tracker.debug = args.verbose
    clones = tracker.run()
    for clone in clones:
        print(clone)
    print(len(clones))
