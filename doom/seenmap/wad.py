"""Read a Doom map from a WAD: vertices, linedefs, sidedefs, sectors, and for the evaluation only the key things.
Used as the reference the seen map is scored against; the agent never reads it (copied from the splat proof of concept, now in the doomsplat repo)."""
import struct
from pathlib import Path

import numpy as np

MAP_LUMPS = ('THINGS', 'LINEDEFS', 'SIDEDEFS', 'VERTEXES', 'SEGS', 'SSECTORS', 'NODES', 'SECTORS', 'REJECT', 'BLOCKMAP')


def freedoom_wad():
    import vizdoom as vzd
    return Path(vzd.__file__).parent / 'freedoom2.wad'


def read_lumps(path, doom_map):
    data = Path(path).read_bytes()
    _, n, off = struct.unpack('<4sII', data[:12])
    entries = [struct.unpack('<II8s', data[off + 16 * i: off + 16 * i + 16]) for i in range(n)]
    names = [e[2].rstrip(b'\0').decode('ascii', 'ignore') for e in entries]
    i = names.index(doom_map)
    lumps = {}
    for j in range(i + 1, min(i + 12, n)):
        if names[j] not in MAP_LUMPS:
            break
        lumps[names[j]] = data[entries[j][0]: entries[j][0] + entries[j][1]]
    return lumps


class MapGeometry:
    def __init__(self, path, doom_map):
        L = read_lumps(path, doom_map)
        self.vertices = np.array(list(struct.iter_unpack('<hh', L['VERTEXES'])), float)
        self.linedefs = np.array(list(struct.iter_unpack('<HHHHHHH', L['LINEDEFS'])), np.int64)  # v1 v2 flags special tag right left
        side = list(struct.iter_unpack('<hh8s8s8sH', L['SIDEDEFS']))
        self.side_sector = np.array([s[5] for s in side], np.int64)
        sec = list(struct.iter_unpack('<hh8s8shhh', L['SECTORS']))
        self.floor = np.array([s[0] for s in sec], float)
        self.ceiling = np.array([s[1] for s in sec], float)
        self.n_sectors = len(sec)
        self.one_sided = self.linedefs[:, 6] == 0xFFFF
        rs, ls = self.linedefs[:, 5], self.linedefs[:, 6]
        rsec = self.side_sector[rs]
        lsec = np.where(self.one_sided, -1, self.side_sector[np.where(self.one_sided, 0, ls)])
        step = ~self.one_sided & ((self.floor[rsec] != self.floor[np.maximum(lsec, 0)]) | (self.ceiling[rsec] != self.ceiling[np.maximum(lsec, 0)]))
        self.wall_lines = self.one_sided | step  # lines that show as a wall or a step from at least one side
        self.right_sector, self.left_sector = rsec, lsec
        self.sector_tag = np.array([s[6] for s in sec], np.int64)
        self.line_special = self.linedefs[:, 3]
        self.line_tag = self.linedefs[:, 4]
        self.things = np.array(list(struct.iter_unpack('<hhhhh', L['THINGS'])), np.int64) if 'THINGS' in L else np.zeros((0, 5), np.int64)

    MANUAL_DOORS = {1, 117, 31, 118, 26, 27, 28, 32, 33, 34}
    REMOTE_DOORS = {2, 109, 86, 106, 4, 108, 90, 105, 3, 16, 110, 75, 76, 107, 103, 112, 61, 115, 29, 111, 63, 114, 50, 113, 42, 116,
                    133, 99, 135, 134, 137, 136}
    KEY_THINGS = {5: 'blue', 40: 'blue', 6: 'yellow', 39: 'yellow', 13: 'red', 38: 'red'}

    def door_sectors(self):
        """Sectors that are doors: the back sector of a manual door line, or the sectors tagged by a remote door line."""
        out = set()
        for k in range(len(self.linedefs)):
            sp = int(self.line_special[k])
            if sp in self.MANUAL_DOORS and not self.one_sided[k]:
                out.add(int(self.left_sector[k]))
            elif sp in self.REMOTE_DOORS and self.line_tag[k] > 0:
                out.update(int(s) for s in np.flatnonzero(self.sector_tag == self.line_tag[k]))
        return out

    def key_markers(self):
        """Key things in THINGS order: (colour, x, y), as the planner's mission data lists them."""
        return [(self.KEY_THINGS[int(t[3])], float(t[0]), float(t[1])) for t in self.things if int(t[3]) in self.KEY_THINGS]

    def segments(self, mask=None):
        ld = self.linedefs if mask is None else self.linedefs[mask]
        return self.vertices[ld[:, 0]], self.vertices[ld[:, 1]]

    def inside(self, pts):
        """Even-odd test of points (N,2) against the one-sided lines: inside the playable map or in the void."""
        a, b = self.segments(self.one_sided)
        x, y = pts[:, 0:1], pts[:, 1:2]
        ax, ay, bx, by = a[:, 0][None], a[:, 1][None], b[:, 0][None], b[:, 1][None]
        cond = (ay > y) != (by > y)
        xint = ax + (y - ay) * (bx - ax) / np.where(by == ay, 1e-9, by - ay)
        return (np.sum(cond & (x < xint), axis=1) % 2) == 1

    def wall_distance(self, pts, chunk=200000):
        """Distance from points (N,2) to the nearest wall or step line."""
        a, b = self.segments(self.wall_lines)
        ab = b - a
        ab2 = np.maximum((ab ** 2).sum(1), 1e-9)
        out = np.empty(len(pts))
        for i in range(0, len(pts), chunk):
            p = pts[i:i + chunk]
            t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(2) / ab2[None], 0, 1)
            proj = a[None] + t[..., None] * ab[None]
            out[i:i + chunk] = np.sqrt(((p[:, None, :] - proj) ** 2).sum(2)).min(1)
        return out

    def sector_at(self, pts):
        """Sector index of each point by the nearest line's facing side (approximate; -1 when in the void)."""
        a, b = self.segments()
        ab = b - a
        ab2 = np.maximum((ab ** 2).sum(1), 1e-9)
        out = np.full(len(pts), -1, np.int64)
        inside = self.inside(pts)
        for i in range(0, len(pts), 50000):
            p = pts[i:i + 50000]
            t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(2) / ab2[None], 0, 1)
            proj = a[None] + t[..., None] * ab[None]
            d2 = ((p[:, None, :] - proj) ** 2).sum(2)
            k = d2.argmin(1)
            cross = ab[k, 0] * (p[:, 1] - a[k, 1]) - ab[k, 1] * (p[:, 0] - a[k, 0])
            sec = np.where(cross < 0, self.right_sector[k], self.left_sector[k])  # right side when the point is to the right of v1->v2
            out[i:i + 50000] = np.where(inside[i:i + 50000], sec, -1)
        return out


if __name__ == '__main__':
    import sys
    g = MapGeometry(freedoom_wad(), sys.argv[1] if len(sys.argv) > 1 else 'MAP01')
    print('vertices', len(g.vertices), 'linedefs', len(g.linedefs), 'one-sided', int(g.one_sided.sum()), 'wall/step lines', int(g.wall_lines.sum()), 'sectors', g.n_sectors)
    print('bbox', g.vertices.min(0), g.vertices.max(0))
    p = np.array([[-192.0, -192.0]])
    s = g.sector_at(p)[0]
    print('start sector', s, 'floor', g.floor[s], 'ceiling', g.ceiling[s], 'inside', g.inside(p)[0], 'wall dist', g.wall_distance(p)[0])
