# Exactly one repair attempt per document

A document that fails validation gets exactly one Repair attempt, receiving every field of each failing rule plus the rule's message; if it still fails, it goes to Human review. Repeated retries push the model to adjust numbers until the arithmetic closes, which produces invented data that passes the validator. Do not raise the attempt count to improve acceptance rates.
