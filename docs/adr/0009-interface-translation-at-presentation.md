---
status: accepted
---

# The demo interface is translated at presentation, keyed by the English messages

The demo interface is Portuguese by default, with English selectable on the page. Everything the visitor reads or downloads is translated: page text, validator messages, upload and limit refusals, gallery captions, field labels, and the Excel workbook. The validator, the service, the API contract, and the prompts stay in English and unchanged. The interface translates validator messages and refusals through a table keyed by the exact English message, with patterns for messages that carry values (item number, CPF or CNPJ, component name). A test produces every message the validator and the refusals can emit and fails when one has no Portuguese translation.

## Considered Options

- **Stable message codes emitted by the validator.** More robust to rewording, but it changes the validation report and the API payload, and the same English messages are what the Repair attempt sends to the model.
- **Fall back to English for an unmapped message.** Never breaks the page, but a new rule would silently ship half-translated.

## Consequences

Rewording a validator message or a refusal breaks the coverage test until its translation is updated. This is intended: the English text is the key.
