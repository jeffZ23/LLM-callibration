"""
Thin wrapper around OpenRouter (OpenAI-compatible API).
Returns parsed (answer_letter, confidence_pct, reasoning) for each call.
"""

import json
import math
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


@dataclass
class LogprobResponse:
    """Token-probability ('internal') confidence for a multiple-choice answer.

    Built from the model's logprobs on the FIRST generated token of a
    letter-only answer: we read the top candidate tokens, keep those that are
    valid option letters, and softmax-normalize over them (Kadavath et al.,
    2022 style). This is the model's intrinsic probability mass on each choice,
    independent of any verbalized self-report.
    """
    answer: str                       # argmax option letter
    prob: float                       # normalized prob of `answer` (0-1)
    dist: dict                        # {letter: normalized prob} over options
    raw_token: str                    # the first generated token (debug)


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


def _norm_letter(token: str) -> str:
    """First alphabetic character of a token, uppercased ('' if none)."""
    for ch in token.strip():
        if ch.isalpha():
            return ch.upper()
    return ""


def call_letter_logprobs(model_id: str, system: str, user: str,
                         option_letters: list[str],
                         max_retries: int = 3,
                         top_logprobs: int = 20) -> LogprobResponse:
    """Ask for a single-letter answer and read the token-probability distribution
    over the option letters from the first token's logprobs.

    Uses temperature=0 so the reported distribution reflects the model's actual
    next-token probabilities. Requires a provider that returns logprobs (OpenAI
    models via OpenRouter do)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = get_client().chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=5,
                logprobs=True,
                top_logprobs=top_logprobs,
            )
            choice = resp.choices[0]
            content = getattr(choice.logprobs, "content", None) if choice.logprobs else None
            if not content:
                raise ValueError("no logprobs returned by provider")

            first = content[0]
            raw_token = first.token
            # Collect logprob for each valid option letter from the top-k
            letter_lp: dict[str, float] = {}
            for cand in first.top_logprobs:
                L = _norm_letter(cand.token)
                if L in option_letters and L not in letter_lp:
                    letter_lp[L] = cand.logprob
            # Ensure the actually-sampled token is represented
            L0 = _norm_letter(raw_token)
            if L0 in option_letters and L0 not in letter_lp:
                letter_lp[L0] = first.logprob

            if not letter_lp:
                raise ValueError(f"no option letter in top tokens: {raw_token!r}")

            # Softmax-normalize over the option letters that appeared
            m = max(letter_lp.values())
            exps = {L: math.exp(lp - m) for L, lp in letter_lp.items()}
            Z = sum(exps.values())
            dist = {L: v / Z for L, v in exps.items()}
            answer = max(dist, key=dist.get)
            return LogprobResponse(answer=answer, prob=dist[answer],
                                   dist=dist, raw_token=raw_token)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"call_letter_logprobs failed after {max_retries} retries: {last_err}")
