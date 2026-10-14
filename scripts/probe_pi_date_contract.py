"""Post-submission diagnostic, not a replacement for immutable baseline verifiers."""

import json

from dates import publication_date

cases = [
    ("valid_alias", "Publicado el 4 de SETIEMBRE de 2016.", "2016-09-04"),
    ("invalid_calendar_first", "30 de febrero de 2016; luego 5 de marzo de 2016", None),
    ("unknown_month_first", "4 de inventado de 2016; luego 5 de febrero de 2016", None),
]
results = []
for name, text, expected in cases:
    try:
        answer = publication_date(text)
        result = {
            "case": name,
            "returned": answer,
            "passed": expected is not None and answer == expected,
        }
    except ValueError:
        result = {"case": name, "raised": "ValueError", "passed": expected is None}
    except Exception as error:
        result = {"case": name, "raised": type(error).__name__, "passed": False}
    results.append(result)
print(json.dumps(results))
