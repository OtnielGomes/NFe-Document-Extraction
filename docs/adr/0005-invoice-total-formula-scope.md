# Invoice total formula covers ICMS-ST but not rarer components

The totals rule is `invoice_total = total_products − discount + icms_st_total + freight + insurance + other_expenses + ipi_total`. ICMS-ST is included because it is common on real invoices and printed on every DANFE. Rarer components of the official vNF formula (import tax, returned IPI, ICMS relief, FCP-ST) are Unsupported components and out of scope, instead of growing the schema.

The Header agent reports which Unsupported components are filled on the DANFE. When any is present, the invoice total rule is skipped and the document goes straight to Human review with a message naming the component, without a Repair attempt: repairing a sum that cannot close only invites the model to adjust numbers until it does.
