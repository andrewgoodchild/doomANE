"""System 2 for doomLaya: the planner's event-driven objective choice asked of a local LLM (Qwen3 8B via MLX).

doom/planner.py asks Laya (System 1) to pick the next objective from code's option list. Here the same
options go to a slower local model instead, with what Laya never sees: the objectives chosen so far and how
each one ended, time on the map, and the reason for re-planning. Laya still answers every 2 Hz command.

The game loop must stay real time, so the LLM is asked in a background thread:
- While an answer is pending, the previous objective stays in force (it is still the model's own choice).
  If that objective just ended (done or failed), there is no objective until the answer arrives, which
  narrows nothing, the same as the planner's own "explore".
- An answer that names no offered option is asked again once with the error; a second bad answer or an
  API error raises, so the run fails instead of falling back to a heuristic.
- An answer that arrives after the options changed (keys, doors, mechanisms) is discarded and re-asked.

Every System 2 question and answer is appended to $PLANNER_LOG with question="system2".

Run from ext/doomLaya-dev with an MLX server up (see docs/LAB_NOTEBOOK.md):
  SYSTEM2_ENDPOINT=http://127.0.0.1:8011/v1/chat/completions .venv/bin/python ../../doom/system2.py <agent.py arguments>
"""
import json
import os
import runpy
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import planner as base  # noqa: E402

ENDPOINT = os.environ.get("SYSTEM2_ENDPOINT", "http://127.0.0.1:8011/v1/chat/completions")
MODEL = os.environ.get("SYSTEM2_MODEL", "mlx-community/Qwen3-8B-4bit")
TIMEOUT = 30
HISTORY = 12  # objectives shown to the model
SYSTEM = ("You are the strategic planner for a bot playing a Doom level. A fast reflex model handles aiming, "
          "fighting and walking; you only choose its next objective. Levels are finished by collecting keys, "
          "using switches that open doors, lower floors or raise lifts, and reaching the exit. "
          "Each option says whether it is reachable now, what it would newly make reachable, and its history. "
          "Prefer options that open something new on the way to missing keys and the exit; do not repeat "
          "objectives that already failed without opening anything; go to the exit as soon as it is reachable. "
          "Weapons and health are side errands: worth it when close or when health is low, but they never finish "
          "a level, and one that was chosen before without being collected is probably out of reach. "
          'Answer with JSON only: {"choice": "<option id>", "why": "<one short sentence>"}.')


def describe(history):
    if not history:
        return "none yet"
    return "; ".join(f"{h['choice']} at {h['at']:.0f}s -> {h.get('outcome', 'in progress')}" for h in history[-HISTORY:])


class System2(base.Planner):
    def reset(self):
        super().reset()
        self.history = []  # choice, at, outcome
        self.pending = None  # thread, result, options, signature, reason
        self.state_text = ""

    def update(self, packet):
        if self.controller is None or self.state is None or self.controller.mission is None:
            return packet
        self.state_text = packet["state"]
        self._collect()
        return super().update(packet)

    def plan(self, state_text, reason, signature):
        """Record how the old objective ended and ask the LLM in the background."""
        if self.history and "outcome" not in self.history[-1]:
            self.history[-1]["outcome"] = reason
        if reason in ("objective done", "objective failed") or reason.startswith("no progress"):
            self.objective = None
        if self.pending is not None:
            return  # an answer is on its way; it is checked against the options when it lands
        options = self.options()
        criteria = {k: v["text"] for k, v in options.items()}
        held = ", ".join(sorted(self.state.get("keys", ()))) or "none"
        user = (f"{self.tick / 35:.0f} s into the level. Health {self.state.get('hp', 0):.0f}. "
                f"Keys held: {held}. Re-planning because: {reason}.\n"
                f"Objectives so far: {describe(self.history)}.\n"
                f"Situation (as the reflex model sees it):\n{state_text}\n\nOptions:\n"
                + "\n".join(f"- {k}: {v}" for k, v in criteria.items()))
        job = dict(options=options, criteria=criteria, signature=signature, reason=reason, user=user,
                   started=time.perf_counter(), tick=self.tick, result=None, error=None)
        job["thread"] = threading.Thread(target=self._run, args=(job,), daemon=True)
        self.pending = job
        job["thread"].start()

    def _run(self, job):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": job["user"]}]
        try:
            for attempt in range(2):
                text = self._chat(messages)
                choice, why, error = self._parse(text, job["criteria"])
                if error is None:
                    job["result"] = dict(choice=choice, why=why, raw=text, attempts=attempt + 1)
                    return
                messages += [{"role": "assistant", "content": text}, {"role": "user", "content": error}]
            job["error"] = f"no valid option after 2 attempts: {text!r}"
        except Exception as e:  # surfaced in the game thread by _collect
            job["error"] = f"{type(e).__name__}: {e}"

    def _chat(self, messages):
        body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": 120, "temperature": 0.0,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        request = urllib.request.Request(ENDPOINT, body, {"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.load(response)["choices"][0]["message"]["content"]

    @staticmethod
    def _parse(text, criteria):
        start, end = text.find("{"), text.rfind("}")
        try:
            answer = json.loads(text[start:end + 1])
        except ValueError:
            return None, None, 'That was not JSON. Answer with JSON only: {"choice": "<option id>", "why": "..."}.'
        choice = str(answer.get("choice", "")).strip()
        if choice not in criteria:
            return None, None, f"{choice!r} is not an option id. Choose one of: {', '.join(criteria)}."
        return choice, str(answer.get("why", "")), None

    def _collect(self):
        job = self.pending
        if job is None or job["thread"].is_alive():
            return
        self.pending = None
        latency = round((time.perf_counter() - job["started"]) * 1000, 1)
        if job["error"]:
            self._log(dict(question="system2", reason=job["reason"], error=job["error"], latency_ms=latency))
            raise RuntimeError(f"System 2 failed: {job['error']}")
        r = job["result"]
        signature = self._signature()
        stale = signature != job["signature"]
        self._log(dict(question="system2", reason=job["reason"], choice=r["choice"], why=r["why"], attempts=r["attempts"],
                       options=job["criteria"], prompt=job["user"], latency_ms=latency, asked_tick=job["tick"],
                       discarded=stale))
        if stale:
            self.plan(self.state_text, "options changed while System 2 was answering", signature)
            return
        if self.history and "outcome" not in self.history[-1]:
            self.history[-1]["outcome"] = "replaced"
        o = job["options"][r["choice"]]
        self.objective = dict(key=r["choice"], kind=o["kind"], target=o["target"], since_tick=self.tick, signature=signature,
                              point=o.get("point"), lift=o.get("lift", False))
        self.rode = False
        self.objective["distance"] = self._distance()
        m = self.memory.setdefault(r["choice"], dict(chosen=0, failed=0, done=False))
        m["chosen"] += 1
        self.history.append(dict(choice=r["choice"], at=self.tick / 35))


def install(endpoint, log_path):
    planner = base.install(endpoint, log_path)
    planner.__class__ = System2
    planner.reset()
    return planner


def main():
    argv = sys.argv[1:]
    endpoint = argv[argv.index("--endpoint") + 1] if "--endpoint" in argv else "http://127.0.0.1:8003/predict"
    if "--mask-unreachable-items" not in argv:
        raise SystemExit("system2.py hooks --mask-unreachable-items; pass it")
    install(endpoint, os.environ.get("PLANNER_LOG"))
    sys.argv = ["agent.py", *argv]
    runpy.run_path(str(Path.cwd() / "agent.py"), run_name="__main__")


if __name__ == "__main__":
    main()
