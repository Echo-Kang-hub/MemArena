from __future__ import annotations

import json
import re
from typing import Any

from app.core.interfaces import EvaluationBench
from app.models.contracts import EvalMetrics, EvalRequest, EvalResult


class LLMJudgeBench(EvaluationBench):
    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    def _fallback_eval(self, request: EvalRequest) -> EvalResult:
        # 当外部模型不可用时，自动回退到规则评估，保障流程稳定
        retrieved_text = "\n".join([hit.content for hit in request.retrieved]).lower()
        expected = [fact.lower() for fact in request.expected_facts]

        if expected:
            matched = sum(1 for fact in expected if fact in retrieved_text)
            precision = min(1.0, matched / max(len(request.retrieved), 1))
            faithfulness = matched / len(expected)
            info_loss = 1.0 - faithfulness
        else:
            precision = 0.7
            faithfulness = 0.7
            info_loss = 0.3

        return EvalResult(
            metrics=EvalMetrics(precision=precision, faithfulness=faithfulness, info_loss=info_loss),
            judge_rationale=(
                "Fallback rule-based judge: provider unavailable or output not parseable. "
                f"precision={precision:.2f}, faithfulness={faithfulness:.2f}, info_loss={info_loss:.2f}"
            ),
            raw_judge_output=None,
        )

    def _extract_json(self, raw_text: str) -> dict[str, Any] | None:
        match = re.search(r"\{[\s\S]*\}", raw_text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def evaluate(self, request: EvalRequest) -> EvalResult:
        judge_prompt = (
            "You are LLM-as-a-Judge for memory retrieval quality."
            " Score based on retrieved memories and expected facts."
            " Return ONLY valid JSON with keys: precision, faithfulness, info_loss, rationale."
            " Scores must be floats in [0,1].\n\n"
            f"Expected Facts: {request.expected_facts}\n"
            f"Retrieved: {[hit.content for hit in request.retrieved]}\n"
            f"Assembled Prompt: {request.assembled_prompt[:1200]}\n"
        )

        raw = self.llm_client.generate(judge_prompt, system_prompt="You are a strict evaluator.")
        parsed = self._extract_json(raw)
        if not parsed:
            return self._fallback_eval(request)

        try:
            precision = float(parsed.get("precision", 0.0))
            faithfulness = float(parsed.get("faithfulness", 0.0))
            info_loss = float(parsed.get("info_loss", 1.0))
            precision = max(0.0, min(1.0, precision))
            faithfulness = max(0.0, min(1.0, faithfulness))
            info_loss = max(0.0, min(1.0, info_loss))
            rationale = str(parsed.get("rationale", "No rationale provided by judge."))
        except (TypeError, ValueError):
            return self._fallback_eval(request)

        return EvalResult(
            metrics=EvalMetrics(precision=precision, faithfulness=faithfulness, info_loss=info_loss),
            judge_rationale=rationale,
            raw_judge_output=raw,
        )
