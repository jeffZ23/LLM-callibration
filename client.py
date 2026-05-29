"""
Thin wrapper around OpenRouter (OpenAI-compatible API).
Returns parsed (answer_letter, confidence_pct, reasoning) for each call.
"""

import json
import os
import re
import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class AgentResponse:
    answer: str          # single letter, e.g. "A"
    confidence: float    # 0-100
    reasoning: str
    raw: str             # raw model output (for the trace / debugging)


_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set OPENROUTER_API_KEY in your environment. "
                "Get a free key at https://openrouter.ai/keys"
            )
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client


_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)
_LETTER = re.compile(r'"?answer"?\s*[:=]\s*"?([A-J])"?', re.IGNORECASE)
_CONF = re.compile(r'"?confidence"?\s*[:=]\s*"?(\d{1,3})"?', re.IGNORECASE)


def _parse(raw: str) -> AgentResponse:
    """Parse {answer, confidence, reasoning} from model output. Robust to
    weak JSON formatting since smaller open models often slip up."""
    answer, confidence, reasoning = None, None, ""

    # First pass: try strict JSON on any {...} block we find.
    for match in _JSON_BLOCK.findall(raw):
        try:
            obj = json.loads(match)
            if "answer" in obj and "confidence" in obj:
                answer = str(obj["answer"]).strip().upper()[:1]
                confidence = float(obj["confidence"])
                reasoning = str(obj.get("reasoning", "")).strip()
                break
        except (json.JSONDecodeError, ValueError):
            continue

    # Fallback: regex
    if answer is None:
        m = _LETTER.search(raw)
        if m:
            answer = m.group(1).upper()
    if confidence is None:
        m = _CONF.search(raw)
        if m:
            confidence = float(m.group(1))

    if answer is None or confidence is None:
        raise ValueError(f"Could not parse answer/confidence from: {raw!r}")

    confidence = max(0.0, min(100.0, confidence))
    return AgentResponse(answer=answer, confidence=confidence,
                         reasoning=reasoning, raw=raw)


def call_agent(model_id: str, system: str, user: str,
               max_retries: int = 3) -> AgentResponse:
    """Call OpenRouter, parse the structured answer, retry on transient failures."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = get_client().chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
                max_tokens=400,
            )
            raw = resp.choices[0].message.content or ""
            return _parse(raw)
        except Exception as e:  # network errors, parse errors, rate limits
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"call_agent failed after {max_retries} retries: {last_err}")
