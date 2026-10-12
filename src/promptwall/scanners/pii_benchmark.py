"""Synthetic benchmark for the PII detector.

Real PII corpora cannot be shared, so this benchmark generates support-ticket style sentences
(English and French) with valid identifiers (correct Luhn, IBAN and NIR checksums), and hard
negatives that look like identifiers but are not: order numbers failing the Luhn check,
invoice references, dates, version strings, prices. Precision and recall are computed on
exact entity spans. It measures the detector's logic, not its coverage of every real-world
format: the limits are stated in the README.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from promptwall.scanners import pii

FIRST = ["alice", "bruno", "chloe", "david", "emma", "farid", "gaelle", "hugo", "ines", "jules"]
LAST = ["martin", "bernard", "dubois", "thomas", "robert", "richard", "petit", "durand"]
DOMAINS = ["example.com", "mail.example.org", "example.fr", "corp.example.net"]


def _luhn_complete(prefix: str, length: int, rng: random.Random) -> str:
    body = prefix + "".join(str(rng.randint(0, 9)) for _ in range(length - len(prefix) - 1))
    for check in range(10):
        if pii.luhn_ok(body + str(check)):
            return body + str(check)
    raise AssertionError("unreachable")


def card(rng: random.Random) -> str:
    number = _luhn_complete(rng.choice(["4", "51", "37"]), rng.choice([16, 16, 15]), rng)
    if rng.random() < 0.5:
        return " ".join(number[i : i + 4] for i in range(0, len(number), 4))
    return number


def iban(rng: random.Random) -> str:
    bban = "".join(str(rng.randint(0, 9)) for _ in range(23))
    numeric = "".join(str(int(c, 36)) for c in bban + "FR00")
    check = 98 - int(numeric) % 97
    value = f"FR{check:02d}{bban}"
    return (
        " ".join(value[i : i + 4] for i in range(0, len(value), 4)) if rng.random() < 0.6 else value
    )


def nir(rng: random.Random) -> str:
    body = (
        f"{rng.choice('12')}{rng.randint(40, 99):02d}{rng.randint(1, 12):02d}"
        f"{rng.randint(1, 95):02d}{rng.randint(1, 999):03d}{rng.randint(1, 999):03d}"
    )
    key = 97 - int(body) % 97
    value = f"{body}{key:02d}"
    if rng.random() < 0.5:
        return f"{value[0]} {value[1:3]} {value[3:5]} {value[5:7]} {value[7:10]} {value[10:13]} {value[13:]}"  # noqa: E501
    return value


def phone(rng: random.Random) -> str:
    digits = [f"{rng.randint(0, 99):02d}" for _ in range(4)]
    style = rng.randint(0, 2)
    if style == 0:
        return f"0{rng.randint(1, 7)} " + " ".join(digits)
    if style == 1:
        return f"+33 {rng.randint(1, 7)} " + " ".join(digits)
    return f"+1 {rng.randint(201, 989)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}"


def email(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)}.{rng.choice(LAST)}{rng.randint(1, 99)}@{rng.choice(DOMAINS)}"


def ip(rng: random.Random) -> str:
    first = rng.choice([o for o in range(11, 224) if o != 127])  # 127/8 is loopback, not PII
    return f"{first}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


GENERATORS = {
    pii.EMAIL: email,
    pii.PHONE: phone,
    pii.CREDIT_CARD: card,
    pii.IBAN: iban,
    pii.FR_NIR: nir,
    pii.IP_ADDRESS: ip,
}

TEMPLATES = {
    pii.EMAIL: [
        "Please send the invoice to {} before Friday.",
        "Mon adresse est {}, merci.",
        "Contact: {} (billing)",
    ],
    pii.PHONE: [
        "You can reach me at {} after 6pm.",
        "Rappelez-moi au {} s'il vous plait.",
        "Phone: {}",
    ],
    pii.CREDIT_CARD: [
        "My card {} was charged twice.",
        "Carte bancaire : {}",
        "Refund to card number {} please.",
    ],
    pii.IBAN: ["Wire the refund to {}.", "Mon IBAN : {}", "IBAN {} (main account)"],
    pii.FR_NIR: ["Numero de securite sociale : {}", "My French social security number is {}."],
    pii.IP_ADDRESS: ["The router at {} keeps rebooting.", "Connexion refusee depuis {}"],
}


def hard_negative(rng: random.Random) -> str:
    """Identifier-like strings that must NOT be flagged."""
    r = rng.randint
    order = "".join(str(r(0, 9)) for _ in range(16))
    choices = [
        f"Order {order} has shipped.",
        f"Invoice INV-{r(2019, 2026)}-{r(10000, 99999)} is attached.",
        f"Firmware {r(1, 9)}.{r(0, 20)}.{r(0, 99)} fixed it.",
        f"Delivered on {r(1, 28):02d}/{r(1, 12):02d}/{r(2020, 2026)}.",
        f"It costs {r(10, 999)}.{r(0, 99):02d} EUR, ref {r(100000, 999999)}.",
        f"Tracking code 1Z{r(100000000, 999999999)}{r(100, 999)}.",
    ]
    return rng.choice(choices)


@dataclass
class Sample:
    text: str
    spans: list[tuple[str, int, int]]


def make_samples(n: int = 2_000, seed: int = 0) -> list[Sample]:
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        if rng.random() < 0.25:
            samples.append(Sample(hard_negative(rng), []))
            continue
        kind = rng.choice(list(GENERATORS))
        value = GENERATORS[kind](rng)
        template = rng.choice(TEMPLATES[kind])
        start = template.index("{}")
        text = template.format(value)
        if kind == pii.CREDIT_CARD and not pii.luhn_ok(value):
            raise AssertionError(value)
        samples.append(Sample(text, [(kind, start, start + len(value))]))
    return samples


def run(n: int = 2_000, seed: int = 0) -> dict[str, Any]:
    tp: dict[str, int] = dict.fromkeys(GENERATORS, 0)
    fp, fn = dict(tp), dict(tp)
    for sample in make_samples(n, seed):
        truth = set(sample.spans)
        found = {(e.kind, e.start, e.end) for e in pii.detect(sample.text)}
        for kind, *_ in found & truth:
            tp[kind] += 1
        for kind, *_ in found - truth:
            fp[kind] += 1
        for kind, *_ in truth - found:
            fn[kind] += 1
    per = {}
    for kind in GENERATORS:
        p = tp[kind] / (tp[kind] + fp[kind]) if tp[kind] + fp[kind] else 1.0
        r = tp[kind] / (tp[kind] + fn[kind]) if tp[kind] + fn[kind] else 1.0
        per[kind] = {"precision": p, "recall": r, "support": tp[kind] + fn[kind]}
    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    return {
        "samples": n,
        "per_entity": per,
        "precision": total_tp / max(total_tp + total_fp, 1),
        "recall": total_tp / max(total_tp + total_fn, 1),
    }
