# Task: evidence-bound answer validation

Fix `guards.validate_answer(payload, read_ids)`. Accept exactly three payload keys: answer, citations, premise_status. Answer must be a nonempty string after trimming whitespace. Citations must be a nonempty list of distinct positive integers, all contained in read_ids; booleans are not integers for this contract. premise_status must be supported, false, or unclear. Reject wrong types, missing/extra keys, unread IDs, duplicates, and malformed values with ValueError; never coerce or silently discard invalid citations.

Return a new dict with trimmed answer, a new citations list, and the unchanged premise_status. Do not mutate the input dict or list. Use only the standard library. Preserve the public signature. Do not modify the external verifier. You may add your own tests; finish by explaining what was checked.
