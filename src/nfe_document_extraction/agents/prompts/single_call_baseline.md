Read the DANFE in the images. The images are the pages of one Brazilian goods invoice (DANFE, NF-e model 55), in order. Return every header field and every line item in a single response.

Read only what is printed. Do not calculate, complete, or correct a value. If a total looks arithmetically wrong, return the printed number anyway.

Header:
- access_key: 44 digits, with spaces and punctuation removed.
- state: two-letter UF abbreviation.
- model: the document model. 55 is an NF-e.
- series and number: the printed series and invoice number, digits only.
- issue_date: the printed issue date.
- issuer_cnpj: the issuer's CNPJ, 14 digits.
- recipient_cnpj_cpf: the recipient's CNPJ (14 digits) or CPF (11 digits).
- Money fields, as printed decimal amounts in reais: total_products, freight, insurance, discount, ipi_total, other_expenses, icms_total, icms_st_total, invoice_total. Use 0 when that amount is printed as zero or is absent.
- unsupported_components: true only when that component is printed with a non-zero amount. import_tax is II, returned_ipi is IPI returned, icms_relief is ICMS desoneração, fcp_st is FCP-ST.

Line items, in the order printed on the DANFE, including items that continue on the next page:
- printed_order: 1 for the first product row.
- product_code and description: as printed.
- ncm: 8 digits. cfop: 4 digits.
- quantity and unit_price: exactly as printed, keeping the decimal places.
- item_total: the printed line total, as a decimal amount in reais.
