# Threat model

PromptWall sits between an LLM application and its model provider. It sees every request and
every response, but nothing else: not the application's retrieval index, not its tools, not
its users' sessions. This page states what that position can and cannot defend.

## Assets

| Asset | Why it matters |
|---|---|
| Users' personal data in prompts | Sent to a third-party provider, possibly logged or retained there |
| Credentials pasted into prompts | Leaked keys are the fastest path to a breach |
| The application's system prompt | Business logic, sometimes secrets; leakage helps attackers craft bypasses |
| Model outputs rendered by the client | Markdown/HTML can trigger requests that exfiltrate conversation data |
| Provider budget and availability | Oversized or flooding requests cost money and capacity |

## Controls and what they cover (OWASP Top 10 for LLM Applications, 2025)

| Control | OWASP | Covers | Does not cover |
|---|---|---|---|
| Injection classifier on `user` and `tool` messages | LLM01 | Known injection and jailbreak styles, direct or carried by tool/RAG content | Novel phrasings (see the cross-source results), non-English text, injections split across turns, image or audio inputs |
| PII pseudonymisation (reversible) | LLM02 | Structured identifiers with checksums (cards, IBAN, NIR), e-mails, phones, IPs | Names, addresses and free-text personal data (needs NER) |
| Secret detection (block) | LLM02 | Documented key formats, private keys, `password=` assignments | Custom or high-entropy secrets without a known format |
| Canary token in the system prompt | LLM07 | Verbatim, spaced, reversed, base64 or hex disclosure of the instructions | Paraphrased or summarised disclosure: the canary is not reproduced |
| Output link and image neutralisation | LLM05 | Zero-click exfiltration through rendered markdown/HTML images | Exfiltration through tool calls the application executes itself |
| Output PII and secret redaction | LLM02 | Identifiers the user did not provide (leaked from context or training data) | Data the user *did* provide being sent somewhere else by a tool |
| Request size limit and rate limiting | LLM10 | Oversized prompts, request floods per client | Distributed abuse across many clients |
| Monitor mode, audit log, metrics | - | Safe rollout, detection engineering, incident review | - |

## Out of scope by design

* **Excessive agency (LLM06).** Tool calls are executed by the application, after the proxy.
  Authorisation (who may refund what, which recipients an e-mail tool accepts) must be
  enforced in code next to the tools; PromptWall only restores placeholders in tool
  arguments.
* **Vector store and supply-chain weaknesses (LLM03, LLM04, LLM08).** The proxy never sees
  documents before they are retrieved, nor the models' provenance.
* **Misinformation (LLM09).** Factual accuracy is not a property a filter can check.

## Residual risks and operating guidance

* A classifier is a probabilistic control. The shipped threshold blocks 1.7% of benign
  in-context questions on the test chatbots (the 1% budget is met on validation chatbots, not
  on unseen ones); tune it on your own traffic in monitor mode before enforcing.
* Expect evasion. The cross-source evaluation shows detection dropping from 99% to 15% on
  attacks written in a style the model never saw: treat the classifier as one layer, never as
  the only one.
* Placeholders change what the model sees. Answers that must reason about the content of an
  e-mail address or a card number (rare in support flows) degrade when PII redaction is on.
* The audit log stores decisions, scores and hashed client identifiers, not prompts. Turning
  `store_text` on makes it a sensitive dataset (pseudonymised, but still a log of
  conversations). It is only honoured for forwarded requests whose user text the guard
  rewrites (enforce mode, PII redaction, secrets blocked or redacted): never for blocked
  requests and never in monitor mode.
