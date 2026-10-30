"""Write the README charts as light and dark SVGs in assets/ (no dependencies).

The numbers are counted from results/doom/ (the runs of the named batch with config planner, or planner_ext_safe
for MAP04 skill 1) and match docs/LAB_NOTEBOOK.md.
    python3 scripts/make_charts.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'assets'
THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', muted='#52514e', grid='#e4e3df', bar='#2a78d6'),
    'dark': dict(surface='#1a1a19', ink='#f0efec', muted='#c3c2b7', grid='#383835', bar='#3987e5'),
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

COMPLETIONS = [  # label, note, completed, runs (results/doom/repeat_<map>_<batch>.json)
    ('MAP01', 'skill 3, exec5', 10, 10),
    ('MAP02', 'skill 3, exec5', 8, 10),
    ('MAP03', 'skill 3, exec5', 4, 10),
    ('MAP04', 'skill 1, skill1 (2 of 3 on GPU)', 3, 3),
    ('MAP04', 'skill 3, all 19 batches', 0, 79),
    ('Held-out MAP05–15', 'skill 3, heldout1, frozen', 0, 18),
]
LATENCY = [  # label, ms per decision (in-process replay of 40 MAP03 packets)
    ('Encoder on ANE, heads on CPU', 612),
    ('+ question heads on ANE', 276),
    ('+ skip one-option questions', 206),
]


def bars(rows, title, subtitle, unit, scale_max, theme, value_text):
    t = THEMES[theme]
    left, right, top, row_h, bar_h = 250, 90, 74, 44, 20
    width = 760
    height = top + row_h * len(rows) + 20
    plot = width - left - right
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'font-family="{FONT}" role="img" aria-label="{title}">',
           f'<rect width="100%" height="100%" rx="8" fill="{t["surface"]}"/>',
           f'<text x="24" y="32" font-size="17" font-weight="600" fill="{t["ink"]}">{title}</text>',
           f'<text x="24" y="54" font-size="13" fill="{t["muted"]}">{subtitle}</text>']
    for frac in (0, .25, .5, .75, 1):
        x = left + plot * frac
        out.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{height - 16}" stroke="{t["grid"]}" stroke-width="1"/>')
    for i, row in enumerate(rows):
        label, note, value = row[0], row[1], row[-1]
        y = top + i * row_h
        w = max(0.0, plot * value / scale_max)
        out.append(f'<text x="{left - 14}" y="{y + (9 if note else 15)}" font-size="14" text-anchor="end" fill="{t["ink"]}">{label}</text>')
        if note:
            out.append(f'<text x="{left - 14}" y="{y + 25}" font-size="11.5" text-anchor="end" fill="{t["muted"]}">{note}</text>')
        if w >= 4:
            # square at the baseline, 4px rounded data end
            r = 4
            out.append(f'<path d="M{left},{y} H{left + w - r} Q{left + w},{y} {left + w},{y + r} V{y + bar_h - r} '
                       f'Q{left + w},{y + bar_h} {left + w - r},{y + bar_h} H{left} Z" fill="{t["bar"]}"/>')
        else:
            out.append(f'<line x1="{left}" y1="{y}" x2="{left}" y2="{y + bar_h}" stroke="{t["muted"]}" stroke-width="2"/>')
        out.append(f'<text x="{left + w + 8}" y="{y + 15}" font-size="13" fill="{t["ink"]}">{value_text(row)}</text>')
    out.append('</svg>')
    return '\n'.join(out) + '\n'


def main():
    OUT.mkdir(exist_ok=True)
    rate = [(label, note, done, runs, 100 * done / runs) for label, note, done, runs in COMPLETIONS]
    lat = [(label, '', ms) for label, ms in LATENCY]
    for theme in THEMES:
        (OUT / f'completions-{theme}.svg').write_text(bars(
            rate, 'Maps completed, real time, every death restarts the map',
            'Runs that reached the exit; budget 3 min for MAP01, 30 min for MAP02, 20 min otherwise', '%', 100, theme,
            lambda r: f'{r[2]}/{r[3]}'))
        (OUT / f'latency-{theme}.svg').write_text(bars(
            lat, 'Milliseconds per decision', 'Replay of 40 recorded MAP03 packets, one process; lower is faster',
            'ms', 800, theme, lambda r: f'{r[-1]} ms'))


if __name__ == '__main__':
    main()
