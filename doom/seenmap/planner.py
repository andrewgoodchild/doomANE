"""Routes and frontiers on the seen map: A* over the occupancy grid (8-connected, no corner cutting), with the cell costs
deciding what may be crossed. Strict planning crosses free cells only; the optimistic variant crosses unknown cells at
double cost; the frontier finder lists free cells next to unknown space.
"""
import heapq
import math

import numpy as np

from mapper import FREE, OCCUPIED, UNKNOWN

INF = float('inf')
MOVES = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]


def cost_grid(state, unknown_cost=INF, passable=None):
    """Cost to enter each cell: free 1, unknown as given, occupied blocked; `passable` (bool grid) forces cells to 1."""
    cost = np.full(state.shape, INF, np.float64)
    cost[state == FREE] = 1.0
    cost[state == UNKNOWN] = unknown_cost
    if passable is not None:
        cost[passable] = 1.0
    return cost


def goal_cells(state, target_ij, radius_cells, cost=None):
    """Cells within the radius of the target that can be entered (free, or passable under the given cost grid)."""
    ti, tj = target_ij
    n = state.shape[0]
    goals = set()
    r = int(math.ceil(radius_cells))
    for i in range(max(0, ti - r), min(n, ti + r + 1)):
        for j in range(max(0, tj - r), min(n, tj + r + 1)):
            if math.hypot(i - ti, j - tj) <= radius_cells:
                ok = (cost[i, j] < INF) if cost is not None else (state[i, j] == FREE)
                if ok:
                    goals.add((i, j))
    return goals


def astar(cost, start, goals, max_expansions=400000):
    """Length in cells of the cheapest path from start to any goal cell, or None. Diagonal moves may not cut a blocked corner."""
    if not goals:
        return None
    if start in goals:
        return 0.0
    n = cost.shape[0]
    gi = np.array([g[0] for g in goals]); gj = np.array([g[1] for g in goals])

    def h(i, j):
        di = np.abs(gi - i); dj = np.abs(gj - j)
        return float(np.min(np.maximum(di, dj) + (math.sqrt(2) - 1) * np.minimum(di, dj)))

    best = {start: 0.0}
    length = {start: 0.0}
    heap = [(h(*start), 0.0, start)]
    expanded = 0
    while heap and expanded < max_expansions:
        f, g, cur = heapq.heappop(heap)
        if g > best.get(cur, INF):
            continue
        if cur in goals:
            return length[cur]
        expanded += 1
        ci, cj = cur
        for di, dj, step in MOVES:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < n and 0 <= nj < n):
                continue
            c = cost[ni, nj]
            if c == INF:
                continue
            if di and dj and (cost[ci + di, cj] == INF or cost[ci, cj + dj] == INF):
                continue
            ng = g + step * c
            if ng < best.get((ni, nj), INF):
                best[(ni, nj)] = ng
                length[(ni, nj)] = length[cur] + step
                heapq.heappush(heap, (ng + h(ni, nj), ng, (ni, nj)))
    return None


def plan(state, start_ij, target_ij, radius_cells=3.0, unknown_cost=INF, passable=None):
    """Path length in cells from the player's cell to within radius of the target, or None."""
    cost = cost_grid(state, unknown_cost, passable)
    if cost[start_ij] == INF:
        cost[start_ij] = 1.0  # the player stands here
    goals = goal_cells(state, target_ij, radius_cells, cost)
    return astar(cost, tuple(start_ij), goals)


def frontiers(state):
    free = state == FREE
    unknown = state == UNKNOWN
    nb = np.zeros_like(unknown)
    nb[1:, :] |= unknown[:-1, :]; nb[:-1, :] |= unknown[1:, :]
    nb[:, 1:] |= unknown[:, :-1]; nb[:, :-1] |= unknown[:, 1:]
    return np.argwhere(free & nb)
