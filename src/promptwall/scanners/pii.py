"""PII detection with checksum validation, and reversible pseudonymisation.

Pattern matching alone over-fires on order numbers and dates, so every structured identifier
that has a checksum is validated (Luhn for card numbers, ISO 13616 mod-97 for IBANs, the
French social-security key for NIRs). Detected values are swapped for stable placeholders
(``[EMAIL_1]``) before the prompt leaves the network, and swapped back in the answer: the
upstream model never sees the raw values but the user experience is unchanged.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

EMAIL = "EMAIL"
PHONE = "PHONE"
CREDIT_CARD = "CREDIT_CARD"
IBAN = "IBAN"
FR_NIR = "FR_NIR"
IP_ADDRESS = "IP_ADDRESS"
ALL_ENTITIES = (EMAIL, PHONE, CREDIT_CARD, IBAN, FR_NIR, IP_ADDRESS)

# When two matches overlap, the more specific entity wins (a NIR key is a 1-in-97 check,
# stronger than Luhn's 1-in-10, so a number passing both is a NIR).
PRIORITY = {IBAN: 0, FR_NIR: 1, CREDIT_CARD: 2, EMAIL: 3, IP_ADDRESS: 4, PHONE: 5}


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def card_ok(number: str) -> bool:
    """Luhn checksum + an issuer prefix (Visa, Mastercard, Amex, Discover, JCB, Diners)."""
    digits = _digits(number)
    prefix_ok = (
        digits[0] == "4"
        or 51 <= int(digits[:2]) <= 55
        or 2221 <= int(digits[:4]) <= 2720
        or digits[:2] in {"34", "37", "35", "36", "38", "30", "65"}
        or digits[:4] == "6011"
    )
    return prefix_ok and luhn_ok(digits)


def luhn_ok(number: str) -> bool:
    digits = [int(d) for d in _digits(number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_ok(iban: str) -> bool:
    compact = re.sub(r"\s", "", iban).upper()
    if not 15 <= len(compact) <= 34 or not compact[:2].isalpha():
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


def nir_ok(nir: str) -> bool:
    """French social security number: 13 digits + 2-digit key (Corsica uses 2A/2B)."""
    compact = re.sub(r"\s", "", nir).upper()
    if len(compact) != 15:
        return False
    body, key = compact[:13], compact[13:]
    body = body.replace("2A", "19").replace("2B", "18")
    if not body.isdigit() or not key.isdigit():
        return False
    return 97 - int(body) % 97 == int(key)


def ip_ok(value: str) -> bool:
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (addr.is_loopback or addr.is_unspecified)


def phone_ok(value: str) -> bool:
    n = len(_digits(value))
    return 9 <= n <= 15


@dataclass(frozen=True)
class Entity:
    kind: str
    start: int
    end: int
    value: str


_RULES: list[tuple[str, re.Pattern[str], Callable[[str], bool]]] = [
    (
        EMAIL,
        re.compile(
            r"(?i)(?<![\w.+-])[a-z0-9][a-z0-9._%+-]*@[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}\b"
        ),
        lambda v: True,
    ),
    (IBAN, re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b"), iban_ok),
    (CREDIT_CARD, re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), card_ok),
    (
        FR_NIR,
        re.compile(
            r"(?<!\d)[12][ ]?\d{2}[ ]?(?:0[1-9]|1[0-2]|[2-9]\d)[ ]?(?:\d{2}|2[AB])"
            r"[ ]?\d{3}[ ]?\d{3}[ ]?\d{2}(?!\d)"
        ),
        nir_ok,
    ),
    (IP_ADDRESS, re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"), ip_ok),
    (
        PHONE,
        re.compile(
            r"(?<![\w+.-])(?:"
            r"(?:\+|00)\d{1,3}[ .-]?(?:\(0\)[ .-]?)?\d{1,4}(?:[ .-]?\d{2,4}){2,4}"  # international
            r"|0[1-9](?:[ .-]?\d{2}){4}"  # French national
            r"|\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}"  # North American
            r")(?![\w.-])"
        ),
        phone_ok,
    ),
]


def detect(text: str, entities: Iterable[str] = ALL_ENTITIES) -> list[Entity]:
    """All non-overlapping PII entities, more specific kinds winning conflicts."""
    wanted = set(entities)
    candidates = [
        Entity(kind, m.start(), m.end(), m.group(0))
        for kind, pattern, valid in _RULES
        if kind in wanted
        for m in pattern.finditer(text)
        if valid(m.group(0))
    ]
    candidates.sort(key=lambda e: (PRIORITY[e.kind], e.start))
    chosen: list[Entity] = []
    for ent in candidates:
        if all(ent.end <= c.start or ent.start >= c.end for c in chosen):
            chosen.append(ent)
    return sorted(chosen, key=lambda e: e.start)


@dataclass
class Vault:
    """Per-request placeholder <-> value mapping (never persisted)."""

    by_value: dict[tuple[str, str], str] = field(default_factory=dict)
    by_placeholder: dict[str, str] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def placeholder(self, kind: str, value: str) -> str:
        key = (kind, value)
        if key not in self.by_value:
            self.counters[kind] = self.counters.get(kind, 0) + 1
            token = f"[{kind}_{self.counters[kind]}]"
            self.by_value[key] = token
            self.by_placeholder[token] = value
        return self.by_value[key]

    def restore(self, text: str) -> str:
        if not self.by_placeholder:
            return text
        pattern = re.compile("|".join(re.escape(p) for p in self.by_placeholder))
        return pattern.sub(lambda m: self.by_placeholder[m.group(0)], text)


def pseudonymise(
    text: str, vault: Vault, entities: Iterable[str] = ALL_ENTITIES
) -> tuple[str, list[Entity]]:
    found = detect(text, entities)
    out, cursor = [], 0
    for ent in found:
        out += [text[cursor : ent.start], vault.placeholder(ent.kind, ent.value)]
        cursor = ent.end
    out.append(text[cursor:])
    return "".join(out), found


def redact(text: str, entities: Iterable[str] = ALL_ENTITIES) -> tuple[str, list[Entity]]:
    """Irreversible masking, used on model output (``[REDACTED_EMAIL]``)."""
    found = detect(text, entities)
    out, cursor = [], 0
    for ent in found:
        out += [text[cursor : ent.start], f"[REDACTED_{ent.kind}]"]
        cursor = ent.end
    out.append(text[cursor:])
    return "".join(out), found
