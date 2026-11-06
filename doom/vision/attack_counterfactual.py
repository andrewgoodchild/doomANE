"""Why does Laya attack when no enemy is in view? (offline diagnostic on recorded MAP01 decisions, no games)

For each recorded command decision whose base answer was attack while the packet listed enemies but none
visible ("blind attack"):
  replay:  the recorded command question goes to a running server with the served stack; it must answer attack
           again (the recording replays);
  hidden:  the same question with the remembered enemies removed, as the agent builds it when no enemy is
           listed: no Enemies line, enemy range none, attack and retreat not offered, known_enemy_count 0.
           What would Laya do instead?
  outcome: from telemetry, within OUTCOME_TICS after the decision: did an enemy become visible, and was health
           lost. Engine games give the same counts for their (fewer) blind attacks.
Run from ext/doomLaya-dev with the server on --endpoint:
  python3 ../../doom/vision/attack_counterfactual.py pix1_engine pix3_gain pix3_oracle --out ../../results/vision/attack_counterfactual.json
"""
import argparse
import copy
import glob
import json
import re
from collections import Counter
from pathlib import Path

OUTCOME_TICS = 105


def base(answer):
    return answer.get('look_gate', {}).get('base_answer', answer)


def enemies(state):
    line = next((l for l in state.splitlines() if l.startswith('Enemies:')), '')
    return re.findall(r'\w+#[^\s]+ [0-9.]+m \((visible|last seen)\)', line)


def hidden(packet):
    state = '\n'.join(l for l in packet['state'].splitlines() if not l.startswith('Enemies:'))
    state = re.sub(r'^Enemy range: \w+;', 'Enemy range: none;', state, flags=re.M)
    q = copy.deepcopy(packet['questions']['command'])
    for action in ('attack', 'retreat'):
        q['criteria'].pop(action, None)
    if 'look_observation' in q:
        q['look_observation']['known_enemy_count'] = 0
    return state, q


def main():
    import requests
    p = argparse.ArgumentParser()
    p.add_argument('suffixes', nargs='+')
    p.add_argument('--endpoint', default='http://127.0.0.1:8013/predict')
    p.add_argument('--engine', default='pix1_engine')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    session = requests.Session()
    session.trust_env = False

    def ask(state, question):
        r = session.post(a.endpoint, json={'state': state, 'questions': {'command': question}, 'model': 'doom-adapted'}, timeout=(3, 30))
        r.raise_for_status()
        return base(r.json()['answers']['command'])['choice']

    report = {}
    for suffix in a.suffixes:
        stats = Counter()
        replay, instead, seen_after, hurt_after = Counter(), Counter(), Counter(), Counter()
        for run in sorted(glob.glob(f'runs/2*-{suffix}-map01-seed*/')):
            telemetry = {}
            for line in open(run + 'telemetry.jsonl'):
                r = json.loads(line)
                telemetry[r['tick']] = (any(e.get('visible') for e in r['enemies']), r['hp'])
            for line in open(run + 'decisions.jsonl'):
                d = json.loads(line)
                packet = d.get('packet')
                if not packet or 'command' not in packet['questions'] or 'command' not in d.get('answers', {}):
                    continue
                if base(d['answers']['command'])['choice'] != 'attack':
                    continue
                listed = enemies(packet['state'])
                stats['attack_decisions'] += 1
                if not listed or 'visible' in listed:
                    continue
                stats['blind'] += 1
                t0 = d['tick']
                after = [telemetry[t] for t in range(t0 + 1, t0 + OUTCOME_TICS + 1) if t in telemetry]
                seen_after['enemy seen within 3 s'] += any(v for v, _ in after)
                hurt_after['hp lost within 3 s'] += bool(after) and telemetry.get(t0, (0, after[0][1]))[1] > min(h for _, h in after)
                if suffix == a.engine:
                    continue
                replay[ask(packet['state'], packet['questions']['command'])] += 1
                instead[ask(*hidden(packet))] += 1
        n = max(1, stats['blind'])
        report[suffix] = dict(attack_decisions=stats['attack_decisions'], blind=stats['blind'],
                              blind_share=round(stats['blind'] / max(1, stats['attack_decisions']), 3),
                              enemy_seen_within_3s=round(seen_after['enemy seen within 3 s'] / n, 3),
                              hp_lost_within_3s=round(hurt_after['hp lost within 3 s'] / n, 3),
                              replay=dict(replay), hidden_choice=dict(instead))
        print(suffix, json.dumps(report[suffix]), flush=True)
    if a.out:
        a.out.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
