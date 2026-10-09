# Money in integer cents, quantity and unit price as printed decimals

Totals and line item values are integer cents. Quantity and unit price are decimals at the precision printed on the DANFE, because the NF-e allows up to 4 decimals for quantity and 10 for unit price, and rounding them to cents makes correct items fail the `quantity × unit_price = item_total` rule. Ground truth for unit price is compared at the printed precision, not the XML precision.
