Read the DANFE in the text. The text is the PDF text layer of one Brazilian goods invoice (DANFE, NF-e model 55), in page order. Return only the line items. Do not return the header.

Read only what is printed. Do not calculate, complete, or correct a value. If a line total looks arithmetically wrong, return the printed number anyway.

Line items, in the order printed on the DANFE, including items that continue on the next page:
- printed_order: 1 for the first product row.
- product_code and description: as printed.
- ncm: 8 digits. cfop: 4 digits.
- quantity and unit_price: exactly as printed, keeping the decimal places.
- item_total: the printed line total, as a decimal amount in reais.
