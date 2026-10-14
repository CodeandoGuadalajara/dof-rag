# Task: Spanish DOF publication date

Fix `dates.publication_date(text)` to return the first Spanish full-month date in text as YYYY-MM-DD. Accept upper/lower case and flexible whitespace, all twelve month names, including both septiembre and setiembre. Example: `Publicado el 4 de febrero de 2016.` -> `2016-02-04`. Validate the actual calendar date, including leap years. If no full date is present, the month is unknown, or the first date is invalid, raise ValueError. Do not silently choose a later date after an invalid first date.

Use only the standard library; no locale installation or network. Preserve the public signature. Do not modify the external verifier. You may add your own tests; finish with a short explanation and what you tested.
