"""Online occupancy map from the player's view: depth buffer plus exact pose, every tic, into a grid of free, wall and
unknown cells. The level's geometry is never read. Calibration and the back-projection come from the splat proof of
concept (2026-11-08, now in the doomsplat repo); the planning structure is the grid, the Gaussian attributes are not needed here.

Camera (half resolution, 200x320 view above the status bar): fx = cx = 160, fy = 195 (fitted on wall bases against the development map, 2026-11-09; was 132), cy = 100. Walls and ceilings:
forward distance = 7.1386 x depth + 14.297 map units (fitted against the development map's one-sided walls,
doomsplat scripts/planes2_depthcal.py; median error 1.7 units at this resolution, against 10 for the earlier 7.6 x depth),
capped below 250. Floors: the depth buffer's floor values follow no scale, so the player's
own floor is recognised by the depth value it shows at each row (FLOOR_DEPTH_BY_ROW) or by the ramp those values follow,
and placed by row geometry at the player's floor height.
"""
import math

import numpy as np

CALIB = dict(fx=160.0, cx=160.0, fy=195.0, cy=100.0, units_per_depth=7.1386, depth_offset=14.297, eye_height=41.0, depth_cap=250,
             weapon_box=(136, 200, 130, 192))  # rows r0:r1, columns c0:c1 of the player's weapon sprite
# the depth value the player's own floor shows at each half-resolution row (doomsplat results/calibration.json); -1 = none
FLOOR_DEPTH_BY_ROW = np.full(200, -1.0)
for _r, _v in {115: 5, 116: 5, 117: 4, 118: 4, 119: 4, 120: 3, 121: 3, 122: 3, 123: 3, 124: 3, 125: 2, 126: 2, 127: 2, 128: 2,
               129: 2, 130: 2, 131: 1, 132: 1, 133: 1, 134: 1, 135: 1, 136: 1, 137: 1, 138: 1, 139: 1, 140: 1, 141: 1, 142: 1,
               171: 194, 172: 194, 173: 194, 174: 194, 175: 194, 176: 194, 177: 193, 178: 193, 179: 193, 180: 193, 181: 193,
               182: 193, 183: 193, 184: 193, 185: 193, 186: 193, 187: 193, 188: 193, 189: 193, 190: 193, 191: 193, 192: 193,
               193: 193, 194: 193, 195: 193, 196: 193, 197: 193, 198: 193, 199: 193}.items():
    FLOOR_DEPTH_BY_ROW[_r] = _v
WALL, FLOOR, CEIL = 0, 1, 2
UNKNOWN, FREE, OCCUPIED = 0, 1, 2


def depth_units(d, C=CALIB):
    """Depth-buffer values -> forward distance in map units; 0 (no reading) stays 0."""
    d = np.asarray(d, np.float64)
    return np.where(d > 0, C['units_per_depth'] * d + C.get('depth_offset', 0.0), 0.0)


def unpack_mask(packed):
    """Sprite mask packed with np.packbits along columns -> bool (200, 320)."""
    return np.unpackbits(packed, axis=1)[:, :320].astype(bool)


def frame_is_inside_geometry(dep, pose):
    """Frames rendered inside a closed door or lift, or with a broken height, carry no view."""
    hz = dep[98:103]
    return np.median(hz) < 3 or (hz == 0).mean() > 0.5 or abs(pose[2]) > 1000


def frame_points(dep, mask, pose, stride=2, C=CALIB, fill=True, below_floor_rule=False, lookup_min_row=115, ramp_min_row=143):
    """Back-project one frame. Returns world points (N, 3) and their class (wall, floor, ceiling); with `fill`, the floor
    between consecutive floor samples of a column is filled in (floor_fill)."""
    fx, cx, fy, cy, s, eye = C['fx'], C['cx'], C['fy'], C['cy'], C['units_per_depth'], C['eye_height']
    r0, r1, c0, c1 = C['weapon_box']
    v, u = np.mgrid[0:200:stride, 0:320:stride]
    d = dep[v, u].astype(np.float64)
    not_weapon = ~((v >= r0) & (v < r1) & (u >= c0) & (u < c1))
    free_pixel = (~mask[v, u]) & not_weapon
    valid = (d > 0) & (d < C['depth_cap']) & free_pixel
    below = v + 0.5 > cy + 1
    look = FLOOR_DEPTH_BY_ROW[v]
    ramp_lo = 45.0 / np.maximum(v + 0.5 - cy - 2, 0.5) - 1.5
    ramp_hi = 80.0 / np.maximum(v + 0.5 - cy - 2, 0.5) + 1.5
    # the player's own floor: the exact value the floor shows at that row (rows 115 on), or, in the rows the lookup does
    # not cover, the near ramp (rows 143 on, depth under 4) and the zero band. Rows above 115 (floor farther than about
    # 350 units) never count as floor: there the ramp also fits walls 200 to 400 units away, which put phantom floor
    # beyond the walls (seen as spikes outside the map in the first full run).
    on_floor = free_pixel & below & (((look >= 0) & (np.abs(d - look) <= 1) & (v >= lookup_min_row))
                                     | ((d >= ramp_lo) & (d <= ramp_hi) & (v >= ramp_min_row) & (v <= 170))
                                     | ((d == 0) & (v >= 135)))
    valid |= on_floor
    z = depth_units(d, C)
    z_floor = fy * eye / np.maximum(v + 0.5 - cy, 1e-3)
    z = np.where(on_floor, z_floor, z)
    horizon = depth_units(np.median(dep[98:103], axis=0)[u], C)
    cap_ok = (horizon <= 0) | (horizon >= depth_units(C['depth_cap'], C)) | (z_floor <= horizon + 16)
    far_floor = on_floor & (~cap_ok | (z_floor > 900))
    below_floor = valid & ~on_floor & ((v + 0.5 - cy) / fy * z > eye + 6)  # computes below the player's floor: a plane pixel
    if below_floor_rule:
        on_floor |= below_floor  # read as the player's own floor with the planes' broken depth (right on a flat floor, wrong over a pit)
        z = np.where(below_floor, z_floor, z)
    else:
        valid &= ~below_floor  # nothing is known about it
    other_flat_below = valid & ~on_floor & (look >= 0) & below & (d < 6)
    valid &= ~other_flat_below & ~far_floor
    X = (u + 0.5 - cx) / fx * z
    Y = -(v + 0.5 - cy) / fy * z
    P = np.stack([X, Y, z], -1)
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    cls = np.where(on_floor, FLOOR, np.where(np.abs(n[..., 1]) > 0.6, np.where(Y > 0, CEIL, FLOOR), WALL))
    jump = (np.abs(np.gradient(z, axis=1)) > 0.08 * z) | (np.abs(np.gradient(z, axis=0)) > 0.08 * z)
    keep = valid & (on_floor | ~jump) & (on_floor | (np.abs(Y) <= 320))
    px, py, pz, ang = pose
    a = np.radians(ang)
    f = np.array([np.cos(a), np.sin(a), 0.0])
    rgt = np.array([np.sin(a), -np.cos(a), 0.0])
    up = np.array([0, 0, 1.0])
    R = np.stack([rgt, up, f], 1)
    W = P[keep] @ R.T + np.array([px, py, pz + eye])
    classes = cls[keep]
    if fill:
        F = floor_fill(on_floor & keep, z_floor, u, v, horizon, stride, C)
        if len(F):
            W = np.concatenate([W, F @ R.T + np.array([px, py, pz + eye])])
            classes = np.concatenate([classes, np.full(len(F), FLOOR)])
    return W, classes


def floor_fill(on_floor, z_floor, u, v, horizon, stride, C=CALIB, step=8.0):
    """The floor between two floor samples in adjacent rows of one column is floor too: one image row covers a long
    stretch of ground at range, so the samples alone leave gaps. Points are added every `step` units along the ground
    ray between consecutive floor rows, and from the farthest floor sample to the wall at the horizon when it is no
    farther than one and a half row gaps away. Returns camera-frame points (right, up, forward) at floor height."""
    fx, cx, fy, cy, eye = C['fx'], C['cx'], C['fy'], C['cy'], C['eye_height']
    rows, cols = on_floor.shape
    pair = on_floor[:-1] & on_floor[1:]  # row r (farther) and row r+1 (nearer) both floor
    r, c = np.nonzero(pair)
    z_far, z_near = z_floor[r, c], z_floor[r + 1, c]
    # the farthest floor sample of each column, extended to the wall at the horizon
    first = np.argmax(on_floor, axis=0)
    has = on_floor.any(axis=0)
    cidx = np.flatnonzero(has)
    zf = z_floor[first[cidx], cidx]
    zw = horizon[0, cidx] if horizon.ndim == 2 else horizon[cidx]
    gap = zf * zf * stride / (fy * eye)
    ext = (zw > zf + step) & (zw <= zf + 1.5 * gap + step) & (zw < depth_units(C['depth_cap'], C))
    z_far = np.concatenate([z_far, (zw[ext] - step)])
    z_near = np.concatenate([z_near, zf[ext]])
    c = np.concatenate([c, cidx[ext]])
    span = z_far - z_near
    ok = span > step
    z_far, z_near, c, span = z_far[ok], z_near[ok], c[ok], span[ok]
    if not len(c):
        return np.zeros((0, 3))
    n = np.ceil(span / step).astype(int)
    idx = np.repeat(np.arange(len(c)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    z = z_near[idx] + (k + 0.5) * span[idx] / n[idx]
    uu = u[0, c[idx]] if u.ndim == 2 else u[c[idx]]
    X = (uu + 0.5 - cx) / fx * z
    return np.stack([X, np.full_like(z, -eye), z], -1)


class OccupancyMap:
    """Votes per cell from wall and floor points, the player's own walked cells, and the three-state view of them."""

    def __init__(self, origin, cell=16.0, size=1024, wall_min=3, wall_ratio=1.0, free_min=2):
        self.cell, self.size = float(cell), int(size)
        self.x0 = float(origin[0]) - cell * size / 2
        self.y0 = float(origin[1]) - cell * size / 2
        self.wall_votes = np.zeros((size, size), np.int32)
        self.free_votes = np.zeros((size, size), np.int32)
        self.walked = np.zeros((size, size), bool)
        self.wall_min, self.wall_ratio, self.free_min = wall_min, wall_ratio, free_min

    def cell_of(self, x, y):
        i = np.floor((np.asarray(x) - self.x0) / self.cell).astype(np.int64)
        j = np.floor((np.asarray(y) - self.y0) / self.cell).astype(np.int64)
        return i, j

    def point_of(self, i, j):
        return self.x0 + (i + 0.5) * self.cell, self.y0 + (j + 0.5) * self.cell

    def _inside(self, i, j):
        return (i >= 0) & (i < self.size) & (j >= 0) & (j < self.size)

    WALL_RANGE = 400.0   # wall points farther than this from the camera are too scattered to vote (splat proof of concept: 27-67 % within 16 units)
    FLOOR_RANGE = 600.0  # floor points farther than this do not vote either
    STEP = 24.0          # Doom's step height: a riser up to this is walked over, not a wall

    def update(self, pts, cls, floor_z, camera_xy=None):
        """Add one frame's points. Walls count from one step above the player's floor to 72 units above it; floors within
        24 of it; both only within range of the camera."""
        z = pts[:, 2]
        wall = (cls == WALL) & (z > floor_z + self.STEP) & (z <= floor_z + 72)
        floor = (cls == FLOOR) & (np.abs(z - floor_z) <= 24)
        if camera_xy is not None:
            dist = np.hypot(pts[:, 0] - camera_xy[0], pts[:, 1] - camera_xy[1])
            wall &= dist <= self.WALL_RANGE
            floor &= dist <= self.FLOOR_RANGE
        for sel, votes in ((wall, self.wall_votes), (floor, self.free_votes)):
            i, j = self.cell_of(pts[sel, 0], pts[sel, 1])
            ok = self._inside(i, j)
            flat = i[ok] * self.size + j[ok]
            if flat.size:
                counts = np.bincount(flat, minlength=self.size * self.size)
                votes += counts.reshape(self.size, self.size).astype(np.int32)

    def walk(self, x, y):
        i, j = self.cell_of(x, y)
        if 0 <= i < self.size and 0 <= j < self.size:
            self.walked[i, j] = True

    def walk_segment(self, x0, y0, x1, y1):
        """The body passed along the segment: every cell it crossed is free (sampled every 4 units, 4-connected)."""
        n = max(int(math.hypot(x1 - x0, y1 - y0) / 4.0), 1)
        xs = np.linspace(x0, x1, n + 1); ys = np.linspace(y0, y1, n + 1)
        i, j = self.cell_of(xs, ys)
        ok = self._inside(i, j)
        self.walked[i[ok], j[ok]] = True
        # 4-connect diagonal steps between consecutive cells
        di, dj = np.diff(i), np.diff(j)
        diag = (di != 0) & (dj != 0)
        for k in np.flatnonzero(diag):
            a, b = i[k] + di[k], j[k]
            if 0 <= a < self.size and 0 <= b < self.size:
                self.walked[a, b] = True

    def state(self):
        """0 unknown, 1 free, 2 wall. A walked cell is free whatever the votes: the player has stood there."""
        wall = (self.wall_votes >= self.wall_min) & (self.wall_votes >= self.wall_ratio * self.free_votes)
        free = self.free_votes >= self.free_min
        s = np.where(wall, OCCUPIED, np.where(free, FREE, UNKNOWN)).astype(np.uint8)
        s[self.walked] = FREE
        return s

    @staticmethod
    def frontier(state):
        """Free cells with an unknown 4-neighbour."""
        free = state == FREE
        unknown = state == UNKNOWN
        nb = np.zeros_like(unknown)
        nb[1:, :] |= unknown[:-1, :]
        nb[:-1, :] |= unknown[1:, :]
        nb[:, 1:] |= unknown[:, :-1]
        nb[:, :-1] |= unknown[:, 1:]
        return free & nb
