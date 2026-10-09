# Line items are aligned by product code and description, not by nItem

The DANFE does not print nItem, so any extracted item number is just the row position, and matching by it makes one skipped row turn every following item into an error. Extracted Line items are paired with Ground truth by an order-preserving sequence alignment over product code and description; an unpaired Ground truth item is one Missing item, an unpaired extracted item is one Spurious item. Product code joins the Line item schema as an Unverifiable field for this purpose.
