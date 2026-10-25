"""A Policy backed by any OpenAI-compatible chat endpoint.

One class for both ends of the project: llama.cpp on a laptop now, SGLang on a
GPU later. Both speak /v1/chat/completions, so nothing downstream changes when
the engine does -- only the base_url.

Stdlib only (urllib), same as everything else the reward path touches.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class APIPolicy:
    """Calls a chat endpoint and returns the completion as raw text.

    `episode.py` stores what this returns verbatim as the assistant message,
    because the phase-2 loss mask covers exactly those spans. So this must
    return what the model actually generated -- see `_raw_text`.
    """

    base_url: str                      # "http://127.0.0.1:8080/v1"
    model: str = "local"
    temperature: float = 1.0           # TMax samples rollouts at 1.0
    max_tokens: int = 512
    timeout: float = 180.0
    retries: int = 3
    api_key: str = "EMPTY"             # llama.cpp ignores it; clients require one
    thinking: bool = True              # False -> Qwen3's non-thinking mode

    def __call__(self, messages: list[dict]) -> str:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if not self.thinking:
            # Measured against llama.cpp --jinja: this suppresses the reasoning
            # block entirely. Note it also makes Qwen3 stop following the output
            # format, so an ablation arm using it needs a tolerant prompt.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return _raw_text(json.loads(r.read()))
            except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
                last = e
                if attempt < self.retries - 1:
                    time.sleep(2 * (attempt + 1))
        # A dead endpoint is our bug, not agent behaviour, so it must not be
        # silently scored as a failed episode.
        raise RuntimeError(f"policy call failed after {self.retries} tries: {last}")


def _raw_text(payload: dict) -> str:
    """Reassemble the completion exactly as the model produced it.

    Not defensive -- REQUIRED. Measured against llama.cpp --jinja serving
    Qwen3-0.6B: a 245-token completion arrived with 535 characters of thinking
    in `reasoning_content` and only `<command>...</command>` in `content`.
    SGLang's --reasoning-parser does the same.

    Taking `content` alone would drop tokens the model really generated, which
    is wrong twice over: the parser would see no <think>, and phase 2 would
    train on a completion missing its reasoning while the loss mask claimed to
    cover it -- training runs, loss falls, nothing improves.
    """
    choice = payload["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if not reasoning or "<think>" in content:
        return content

    # Only close the tag if generation actually finished. A completion cut off
    # by max_tokens stops mid-thought, and fabricating </think> would make it
    # look complete -- the parser's "unterminated <think>" diagnosis exists
    # precisely to tell "ran out of room" apart from "produced no action", and
    # those call for opposite fixes (raise max_tokens vs change the prompt).
    if choice.get("finish_reason") == "length":
        return f"<think>{reasoning}{content}"
    return f"<think>{reasoning}</think>{content}"
