"""
Backtracking reference solver / validator for the pairwise triplet scheduling problem.

Problem: n students (n divisible by 3). Each quiz ("round") partitions the students
into n/3 groups of 3. No pair of students may be grouped together more than once
across all rounds. Maximise the number of rounds.

This module provides:
  * validate_schedule  - checks any schedule (e.g. output of YOUR algorithm)
  * find_schedule      - exhaustive backtracking: finds a schedule with `rounds` rounds, or None
  * max_rounds         - exact optimum by backtracking (small n only)
  * check_algorithm    - validates an algorithm's schedule and compares to the optimum

Students are labelled 0..n-1. A schedule is a list of rounds; a round is a list of
3-tuples, e.g. [[(0,1,2),(3,4,5)], [(0,3,x), ...]].
"""

from itertools import combinations
import sys
import time


# --------------------------------------------------------------------------- #
# Basics
# --------------------------------------------------------------------------- #
def get_pairs(group) -> list[frozenset]:
    """The three unordered pairs inside a group of three students."""
    a, b, c = group
    return [frozenset((a, b)), frozenset((b, c)), frozenset((a, c))]


def theoretical_upperbound(n: int) -> int:
    """floor((n-1)/2): total pairs n(n-1)/2 divided by n pairs consumed per round."""
    return (n - 1) // 2


# --------------------------------------------------------------------------- #
# Validator (use this on the output of your constructive algorithm)
# --------------------------------------------------------------------------- #
def validate_schedule(n: int, schedule) -> tuple[bool, str]:
    """Return (is_valid, message). Checks every constraint of the problem."""
    if n % 3 != 0:
        return False, f"n={n} is not divisible by 3"

    students = set(range(n))
    seen_pairs = set()

    for r, rnd in enumerate(schedule, start=1):
        if len(rnd) != n // 3:
            return False, f"Round {r}: expected {n // 3} groups, got {len(rnd)}"

        in_round = []
        for g in rnd:
            if len(g) != 3 or len(set(g)) != 3:
                return False, f"Round {r}: group {g} is not 3 distinct students"
            in_round.extend(g)

        if set(in_round) != students or len(in_round) != n:
            return False, f"Round {r}: groups are not a partition of all {n} students"

        for g in rnd:
            for p in get_pairs(g):
                if p in seen_pairs:
                    a, b = sorted(p)
                    return False, f"Round {r}: pair ({a},{b}) was already used earlier"
                seen_pairs.add(p)

    return True, f"Valid: {len(schedule)} rounds, {len(seen_pairs)} distinct pairs used"


# --------------------------------------------------------------------------- #
# Backtracking search
# --------------------------------------------------------------------------- #
def find_schedule(n: int, rounds: int):
    """
    Exhaustively search for a schedule with exactly `rounds` rounds.
    Returns the schedule (list of rounds) or None if none exists.

    Pruning / symmetry breaking (all lossless, so a None answer is a proof):
      * Within a round, always place the smallest unassigned student next, so each
        partition is generated once (no reordering of groups).
      * Round 1 is fixed to {0,1,2},{3,4,5},... (students are interchangeable).
      * Rounds are interchangeable, so the group containing student 0 must be
        strictly increasing (lexicographically) from round to round.
    """
    if n % 3 != 0:
        raise ValueError("n must be divisible by 3")
    if rounds <= 0:
        return []
    if rounds > theoretical_upperbound(n):
        return None  # violates the proven upper bound

    used = [[False] * n for _ in range(n)]  # used[a][b]: a and b already grouped
    schedule: list[list[tuple]] = []

    def mark(g, value):
        a, b, c = g
        used[a][b] = used[b][a] = value
        used[b][c] = used[c][b] = value
        used[a][c] = used[c][a] = value

    def build_round(groups, assigned) -> bool:
        # Find the smallest unassigned student.
        first = next((i for i in range(n) if not assigned[i]), None)

        if first is None:  # round complete
            schedule.append(list(groups))
            if len(schedule) == rounds or build_round([], [False] * n):
                return True
            schedule.pop()
            return False

        first_round = len(schedule) == 0
        prev_key = schedule[-1][0] if schedule else None  # group containing 0

        for b in range(first + 1, n):
            if assigned[b] or used[first][b]:
                continue
            for c in range(b + 1, n):
                if assigned[c] or used[first][c] or used[b][c]:
                    continue

                g = (first, b, c)

                # Symmetry breaking.
                if first_round and g != (first, first + 1, first + 2):
                    continue
                if first == 0 and prev_key is not None and g <= prev_key:
                    continue

                mark(g, True)
                assigned[first] = assigned[b] = assigned[c] = True
                groups.append(g)

                if build_round(groups, assigned):
                    return True

                groups.pop()
                assigned[first] = assigned[b] = assigned[c] = False
                mark(g, False)

        return False

    sys.setrecursionlimit(10_000)
    return schedule if build_round([], [False] * n) else None


def max_rounds(n: int):
    """
    Exact optimum by backtracking: try the upper bound first and go down until a
    schedule exists. Returns (max_rounds, schedule). Practical for small n
    (roughly n <= 15 or 18); infeasibility proofs grow quickly.
    """
    for q in range(theoretical_upperbound(n), 0, -1):
        sched = find_schedule(n, q)
        if sched is not None:
            return q, sched
    return 0, []


# --------------------------------------------------------------------------- #
# Compare YOUR algorithm against the backtracking ground truth
# --------------------------------------------------------------------------- #
def check_algorithm(n: int, algo_schedule, compute_optimum: bool = True) -> bool:
    """
    1. Validates the algorithm's schedule.
    2. Compares its round count to the backtracking optimum (and the upper bound).
    Returns True if the schedule is valid and (when computed) optimal.
    """
    ok, msg = validate_schedule(n, algo_schedule)
    print(f"[n={n}] algorithm: {msg}")
    if not ok:
        return False

    bound = theoretical_upperbound(n)
    got = len(algo_schedule)
    print(f"[n={n}] theoretical upper bound: {bound}, algorithm achieved: {got}")

    if not compute_optimum:
        return True

    best, _ = max_rounds(n)
    print(f"[n={n}] backtracking optimum: {best}")
    if got < best:
        print(f"[n={n}] SUBOPTIMAL: algorithm is {best - got} round(s) short")
        return False
    if got > best:
        print(f"[n={n}] MISMATCH: algorithm beats the exhaustive optimum "
              f"(validator or search has a bug)")
        return False
    return True


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def print_schedule(schedule):
    for r, rnd in enumerate(schedule, start=1):
        print(f"  Quiz {r}: " + "  ".join(str(set(g)) for g in rnd))


if __name__ == "__main__":
    sizes = [int(x) for x in sys.argv[1:]] or [3, 6, 9, 12, 15]
    for n in sizes:
        t0 = time.time()
        best, sched = max_rounds(n)
        print(f"n={n:>3}  upper bound={theoretical_upperbound(n):>2}  "
              f"exact max={best:>2}  ({time.time() - t0:.2f}s)")
        print_schedule(sched)
        print(" ", validate_schedule(n, sched)[1])
        print()