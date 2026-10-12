"""Canary tokens: a tripwire for system-prompt leakage.

The proxy plants a random token in the system message of every request. The token has no
meaning for the model, so if it ever appears in an answer, the model has disclosed its
instructions. Disclosure is often obfuscated (spaced out, reversed, base64 or hex encoded),
so the check normalises and decodes the output before matching.
"""

from __future__ import annotations

import base64
import binascii
import re
import secrets

PREFIX = "pw"
_B64 = re.compile(r"[A-Za-z0-9+/_-]{16,}={0,2}")
_HEX = re.compile(r"(?:[0-9a-fA-F]{2}[\s:]?){12,}")


def new_canary() -> str:
    return f"{PREFIX}{secrets.token_hex(8)}"


def plant(system_prompt: str, canary: str) -> str:
    """Attach the canary as an inert comment at the top of the system prompt."""
    return f"<!-- {canary} -->\n{system_prompt}" if system_prompt else f"<!-- {canary} -->"


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _decoded_candidates(text: str) -> list[str]:
    out = []
    for m in _B64.finditer(text):
        chunk = m.group(0)
        for variant in (chunk, chunk + "=" * (-len(chunk) % 4)):
            try:
                out.append(
                    base64.b64decode(variant, altchars=b"-_", validate=False).decode("latin-1")
                )
                break
            except (binascii.Error, ValueError):
                continue
    for m in _HEX.finditer(text):
        try:
            out.append(bytes.fromhex(re.sub(r"[\s:]", "", m.group(0))).decode("latin-1"))
        except ValueError:
            continue
    return out


def leaked(output: str, canary: str) -> str | None:
    """How the canary leaked ("plain", "obfuscated", "reversed", "encoded"), or None."""
    if canary in output:
        return "plain"
    target = _normalise(canary)
    flat = _normalise(output)
    if target in flat:
        return "obfuscated"
    if target[::-1] in flat:
        return "reversed"
    if any(target in _normalise(decoded) for decoded in _decoded_candidates(output)):
        return "encoded"
    return None
