from __future__ import annotations

import json
import re
from typing import Any

from app.core.interfaces import EvaluationBench
from app.models.contracts import EvalMetrics, EvalRequest, EvalResult


class LLMJudgeBench(EvaluationBench):
    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    REFUSAL_PATTERNS = [
        r"不知道",
        r"不清楚",
        r"无法确定",
        r"无法回答",
        r"不能确定",
        r"无从判断",
        r"没有(足够|相关)?信息",
        r"未提供(相关)?信息",
        r"i don't know",
        r"i do not know",
        r"cannot determine",
        r"unable to answer",
        r"insufficient information",
        r"not enough information",
    ]

    NON_REFUSAL_PATTERNS = [
        r"已记住",
        r"我记得",
        r"你提到",
        r"根据(你|上述|已知)",
        r"结论是",
        r"答案是",
        r"可以确认",
        r"已确认",
        r"建议",
        r"如下",
    ]

    def _is_refusal(self, response: str, expected_facts: list[str] | None = None) -> bool:
        lowered = (response or "").strip().lower()

        # 空回答视为拒答（abstain）。
        if not lowered:
            return True

        # 反例库：出现明显“有依据回答”的模式，直接判非拒答。
        if any(re.search(pattern, lowered) for pattern in self.NON_REFUSAL_PATTERNS):
            return False

        # 保护：若输出已包含期望事实，倾向判为非拒答。
        expected = self._normalize_facts(expected_facts or [])
        if expected and any(fact in lowered for fact in expected):
            return False

        refusal_hits = sum(1 for pattern in self.REFUSAL_PATTERNS if re.search(pattern, lowered))
        return refusal_hits > 0

    @staticmethod
    def _normalize_facts(facts: list[str]) -> list[str]:
        return [fact.strip().lower() for fact in facts if str(fact).strip()]

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _compute_recall_at_k(self, request: EvalRequest) -> float:
        expected = self._normalize_facts(request.expected_facts)
        if not expected:
            return 1.0
        retrieved_text = "\n".join(hit.content for hit in request.retrieved).lower()
        matched = sum(1 for fact in expected if fact in retrieved_text)
        return self._clamp01(matched / len(expected))

    def _compute_qa_accuracy(self, request: EvalRequest) -> float:
        expected = self._normalize_facts(request.expected_facts)
        if not expected:
            return 1.0
        answer = (request.generated_response or "").lower()
        return 1.0 if all(fact in answer for fact in expected) else 0.0

    def _compute_qa_f1(self, request: EvalRequest) -> float:
        # 事实级 QA F1：以 expected_facts 为目标单元，稳定且可解释。
        expected = self._normalize_facts(request.expected_facts)
        if not expected:
            return 1.0

        answer = (request.generated_response or "").lower()
        matched = sum(1 for fact in expected if fact in answer)
        if matched == 0:
            return 0.0

        # 在无标准答案文本时，用 expected_facts 命中率近似 QA F1。
        precision = matched / len(expected)
        recall = matched / len(expected)
        if precision + recall == 0:
            return 0.0
        return self._clamp01((2 * precision * recall) / (precision + recall))

    def _compute_consistency_score(self, request: EvalRequest) -> float:
        expected = self._normalize_facts(request.expected_facts)
        answer = (request.generated_response or "").lower()

        if not expected:
            return 1.0 if self._is_refusal(answer) else 0.8

        affirmed = 0
        contradictions = 0
        for fact in expected:
            if fact in answer:
                affirmed += 1
            neg_pattern = rf"(不|没|无|未|不是|并非)[^。；，,\n]{{0,8}}{re.escape(fact)}"
            if re.search(neg_pattern, answer):
                contradictions += 1

        base = affirmed / len(expected)
        penalty = contradictions / len(expected)
        return self._clamp01(base * (1 - penalty))

    def _compute_rejection_rate(self, request: EvalRequest) -> float:
        actual_reject = self._is_refusal(request.generated_response or "", request.expected_facts)
        return 1.0 if actual_reject else 0.0

    def _compute_rejection_correctness_unknown(self, request: EvalRequest) -> float | None:
        expected = self._normalize_facts(request.expected_facts)
        if expected:
            return None
        actual_reject = self._is_refusal(request.generated_response or "", request.expected_facts)
        return 1.0 if actual_reject else 0.0

    def _fallback_eval(self, request: EvalRequest) -> EvalResult:
        # 当外部模型不可用时，自动回退到规则评估，保障流程稳定
        retrieved_text = "\n".join([hit.content for hit in request.retrieved]).lower()
        expected = [fact.lower() for fact in request.expected_facts]
        recall_at_k = self._compute_recall_at_k(request)
        qa_accuracy = self._compute_qa_accuracy(request)
        qa_f1 = self._compute_qa_f1(request)
        consistency_score = self._compute_consistency_score(request)
        rejection_rate = self._compute_rejection_rate(request)
        rejection_correctness_unknown = self._compute_rejection_correctness_unknown(request)

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
            metrics=EvalMetrics(
                precision=precision,
                faithfulness=faithfulness,
                info_loss=info_loss,
                recall_at_k=recall_at_k,
                qa_accuracy=qa_accuracy,
                qa_f1=qa_f1,
                consistency_score=consistency_score,
                rejection_rate=rejection_rate,
                rejection_correctness_unknown=rejection_correctness_unknown,
            ),
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
            f"Generated Response: {request.generated_response[:1200]}\n"
            f"Assembled Prompt: {request.assembled_prompt[:1200]}\n"
        )

        recall_at_k = self._compute_recall_at_k(request)
        qa_accuracy = self._compute_qa_accuracy(request)
        qa_f1 = self._compute_qa_f1(request)
        consistency_score = self._compute_consistency_score(request)
        rejection_rate = self._compute_rejection_rate(request)
        rejection_correctness_unknown = self._compute_rejection_correctness_unknown(request)

        raw = self.llm_client.generate(
            judge_prompt,
            system_prompt="You are a strict evaluator.",
            purpose="judge",
        )
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
            metrics=EvalMetrics(
                precision=precision,
                faithfulness=faithfulness,
                info_loss=info_loss,
                recall_at_k=recall_at_k,
                qa_accuracy=qa_accuracy,
                qa_f1=qa_f1,
                consistency_score=consistency_score,
                rejection_rate=rejection_rate,
                rejection_correctness_unknown=rejection_correctness_unknown,
            ),
            judge_rationale=rationale,
            raw_judge_output=raw,
        )
