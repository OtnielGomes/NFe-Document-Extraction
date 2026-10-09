# NFe Document Extraction

Turns a printed Brazilian goods invoice into structured data and states, field by field, whether the reading can be trusted. The model reads; deterministic rules decide.

## Documents

**NF-e**:
The Brazilian electronic invoice for goods, legally defined by its XML.
_Avoid_: Nota, invoice XML

**DANFE**:
The printed or PDF representation of an NF-e. The only document type this project reads.
_Avoid_: Invoice PDF, receipt

**Rendition**:
One concrete form in which a DANFE reaches the system: a digital PDF, a Simulated photo, or a Phone photo.
_Avoid_: Input file, version

**Simulated photo**:
A Rendition produced by programmatically degrading a digital PDF (perspective, blur, shadow, compression). Cleaner than a real photo.
_Avoid_: Fake photo, scan

**Phone photo**:
A Rendition captured by a real phone camera from a printed DANFE.
_Avoid_: Picture, scan

**Access key**:
The 44-digit identifier of an NF-e (chave de acesso), which encodes state, year/month, issuer CNPJ, model, series, number, and a check digit.
_Avoid_: Chave, invoice key, ID

**Header**:
The invoice-level fields of a DANFE: identification, parties, and totals.
_Avoid_: Metadata, summary

**Line item**:
One product row of a DANFE, identified by its printed order. The DANFE does not print nItem.
_Avoid_: Product, row, item line

## Reading

**Extraction**:
The model's reading of a Rendition into Header and Line item fields. Never authoritative on its own.
_Avoid_: Parsing, OCR result

**Extraction approach**:
A specific combination of input modality and model used to produce an Extraction (cheap vision, strong vision, text layer + text model).
_Avoid_: Strategy, pipeline variant

**Section agent**:
An agent responsible for extracting one section of the DANFE: the Header agent or the Line items agent. Each sees the whole document and owns only its section's fields.
_Avoid_: Worker, sub-agent

**Single-call baseline**:
An Extraction produced by one model call reading every section at once; the reference that Section agents must beat.
_Avoid_: Monolithic agent, one-shot extraction

**Unsupported component**:
A total component printed on the DANFE that the invoice total rule does not model (import tax, returned IPI, ICMS relief, FCP-ST). Its presence sends the document straight to Human review.
_Avoid_: Exotic tax, unknown field

## Trust

**Validation rule**:
A deterministic check over one or more fields that either passes or fails with a message. Rules apply equally to model output and human corrections.
_Avoid_: Check, constraint, guardrail

**Verified field**:
A field covered by at least one Validation rule, with all covering rules passing.
_Avoid_: Valid field, confirmed field

**Unverifiable field**:
A field no Validation rule covers (for example, a line item description). Accepted but reported as unverified.
_Avoid_: Unchecked field, optional field

**Flagged field**:
A field that takes part in at least one failing Validation rule. When a sum fails, every term of the sum is flagged.
_Avoid_: Invalid field, error field

**Accepted document**:
A document with no Flagged fields. It may still contain Unverifiable fields.
_Avoid_: Valid document, approved document

**Rejected document**:
A Rendition in which the model found no Access key. It is not treated as a DANFE: no Repair attempt and no Human review.
_Avoid_: Invalid document, failed document

**Repair attempt**:
The single re-extraction of the Flagged fields, given the failing rules' messages. Exactly one per document.
_Avoid_: Retry, correction loop

**Human review**:
The step where a person edits only the Flagged fields of a document that still fails after the Repair attempt; the edits are validated again.
_Avoid_: Manual check, approval

**Visitor feedback**:
A demo visitor's thumbs up or down on the result of one Extraction, with an optional short comment. Unlike Human review, it changes no field and is never validated.
_Avoid_: Rating, review, user feedback

## Evaluation

**Ground truth**:
The value of every field as printed on the DANFE, derived from the NF-e XML. For a document with a Planted error, it is the erroneous printed value.
_Avoid_: Label, expected output

**Synthetic set**:
Fictitious NF-e generated and rendered by the project; published.
_Avoid_: Fake data, test set

**Real set**:
The author's own purchase invoices from varied issuing systems; only aggregated results are published.
_Avoid_: Production data, private set

**Planted error**:
A deliberate inconsistency inserted into a synthetic DANFE to prove that a Validation rule catches it. Measured by detection rate, never by field accuracy.
_Avoid_: Injected bug, corrupted sample

**Item alignment**:
The order-preserving pairing of extracted Line items with Ground truth Line items, by product code and description.
_Avoid_: Item matching by position

**Missing item**:
A Ground truth Line item with no aligned extracted item. Counts as one error.
_Avoid_: Skipped row

**Spurious item**:
An extracted Line item with no aligned Ground truth item. Counts as one error.
_Avoid_: Hallucinated row, extra item

**Perfect invoice**:
A document whose Extraction matches Ground truth in every field.
_Avoid_: Clean extraction, 100% document
