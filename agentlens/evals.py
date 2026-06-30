"""
Eval harness.

Declarative, regression-style checks you run against a completed Trace. This is
the "eval-first" layer: encode the behavior you require and fail loudly when an
agent drifts.

    from agentlens import evals as E

    suite = E.Suite("support-agent guardrails", [
        E.succeeded(),
        E.called_tool("lookup_order"),
        E.never_called_tool("issue_refund"),
        E.max_cost(0.05),
        E.max_duration_ms(5000),
        E.tool_before("lookup_order", "send_reply"),
        E.custom("answer mentions order", lambda t: any(
            "order" in str(s.outputs).lower() for s in t.spans)),
    ])
    report = suite.run(trace)
    print(report.summary())
    assert report.passed
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .trace import Trace, Span


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Report:
    suite: str
    results: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def num_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def summary(self) -> str:
        lines = [f"Eval suite: {self.suite}",
                 f"  {self.num_passed}/{len(self.results)} checks passed"]
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            extra = f" — {r.detail}" if r.detail else ""
            lines.append(f"  [{mark}] {r.name}{extra}")
        return "\n".join(lines)


# A Check is a function: Trace -> CheckResult
Check = Callable[[Trace], CheckResult]


def _tools(trace: Trace) -> List[Span]:
    return [s for s in trace.spans if s.kind == "tool"]


# ---- built-in checks ---------------------------------------------------
def succeeded() -> Check:
    def _c(t: Trace) -> CheckResult:
        ok = t.status == "ok"
        return CheckResult("run succeeded", ok, "" if ok else f"status={t.status}")
    return _c


def called_tool(name: str) -> Check:
    def _c(t: Trace) -> CheckResult:
        names = [s.name for s in _tools(t)]
        ok = name in names
        return CheckResult(f"called tool '{name}'", ok,
                           "" if ok else f"tools seen: {names}")
    return _c


def never_called_tool(name: str) -> Check:
    def _c(t: Trace) -> CheckResult:
        names = [s.name for s in _tools(t)]
        ok = name not in names
        return CheckResult(f"never called tool '{name}'", ok,
                           "" if ok else f"forbidden tool was called")
    return _c


def tool_before(first: str, second: str) -> Check:
    """Assert `first` tool ran before `second` (ordering guardrail)."""
    def _c(t: Trace) -> CheckResult:
        order = [s.name for s in _tools(t)]
        ok = first in order and second in order and order.index(first) < order.index(second)
        return CheckResult(f"'{first}' before '{second}'", ok,
                           "" if ok else f"order: {order}")
    return _c


def max_cost(usd: float) -> Check:
    def _c(t: Trace) -> CheckResult:
        ok = t.total_cost_usd <= usd
        return CheckResult(f"cost <= ${usd}", ok,
                           f"actual=${t.total_cost_usd:.6f}")
    return _c


def max_tokens(n: int) -> Check:
    def _c(t: Trace) -> CheckResult:
        ok = t.total_tokens <= n
        return CheckResult(f"tokens <= {n}", ok, f"actual={t.total_tokens}")
    return _c


def max_duration_ms(ms: float) -> Check:
    def _c(t: Trace) -> CheckResult:
        actual = t.duration_ms or 0.0
        ok = actual <= ms
        return CheckResult(f"duration <= {ms}ms", ok, f"actual={actual:.1f}ms")
    return _c


def had_event(name: str) -> Check:
    """Assert a marker event was recorded (e.g. 'human_approval')."""
    def _c(t: Trace) -> CheckResult:
        ok = any(s.kind == "event" and s.name == name for s in t.spans)
        return CheckResult(f"event '{name}' occurred", ok)
    return _c


def custom(name: str, predicate: Callable[[Trace], bool]) -> Check:
    def _c(t: Trace) -> CheckResult:
        try:
            ok = bool(predicate(t))
            return CheckResult(name, ok)
        except Exception as exc:
            return CheckResult(name, False, f"raised {exc}")
    return _c


def _trace_transcript(trace: Trace) -> str:
    """Render a trace's spans into a compact transcript for an LLM judge."""
    lines = [f"TASK/METADATA: {trace.metadata}"]
    for s in trace.spans:
        piece = f"[{s.kind}] {s.name}"
        if s.inputs:
            piece += f" in={s.inputs}"
        if s.outputs:
            piece += f" out={s.outputs}"
        if s.status == "error":
            piece += f" ERROR={s.error}"
        lines.append(piece)
    return "\n".join(lines)


def llm_judge(rubric: str, *, client: Any = None, model: Optional[str] = None,
              threshold: float = 0.7, name: Optional[str] = None) -> Check:
    """LLM-as-judge check: score a trace against a natural-language rubric.

    Uses a Groq client by default (fast + cheap). The judge must return JSON
    {"score": 0..1, "reason": "..."}; the check passes when score >= threshold.

        E.llm_judge("The reply correctly answers the order-status question "
                    "and does not invent information.")

    Pass your own `client` (anything with a `.complete(prompt, system=...)`
    returning an object with `.text`) to use a different provider.
    """
    check_name = name or "llm judge"

    def _c(t: Trace) -> CheckResult:
        judge = client
        if judge is None:
            try:
                from .providers.groq import GroqClient, GroqError
                judge = GroqClient(model=model or "llama-3.3-70b-versatile")
            except Exception as exc:  # import-time issues
                return CheckResult(check_name, False, f"judge unavailable: {exc}")

        system = (
            "You are a strict evaluator of AI agent runs. Given a rubric and a "
            "transcript of the agent's steps, respond with ONLY a JSON object: "
            '{"score": <float 0..1>, "reason": "<short>"}. No prose, no markdown.'
        )
        prompt = (
            f"RUBRIC:\n{rubric}\n\nAGENT TRANSCRIPT:\n{_trace_transcript(t)}\n\n"
            "Return the JSON now."
        )
        try:
            resp = judge.complete(prompt, system=system, temperature=0.0)
            text = resp.text.strip()
            # tolerate accidental code fences
            if text.startswith("```"):
                text = text.strip("`")
                text = text[text.find("{"):]
            parsed = json.loads(text[text.find("{"): text.rfind("}") + 1])
            score = float(parsed.get("score", 0.0))
            reason = str(parsed.get("reason", ""))[:160]
            ok = score >= threshold
            return CheckResult(check_name, ok, f"score={score:.2f} (>= {threshold}) — {reason}")
        except Exception as exc:
            return CheckResult(check_name, False, f"judge error: {exc}")

    return _c


@dataclass
class Suite:
    name: str
    checks: List[Check]

    def run(self, trace: Trace) -> Report:
        return Report(self.name, [c(trace) for c in self.checks])
