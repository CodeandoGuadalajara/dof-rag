import copy
import unittest

from guards import validate_answer


class CitationTests(unittest.TestCase):
    def test_valid_and_no_mutation(self):
        for status in ["supported", "false", "unclear"]:
            payload = {
                "answer": "  respuesta  ",
                "citations": [101, 202],
                "premise_status": status,
            }
            original = copy.deepcopy(payload)
            result = validate_answer(payload, {101, 202})
            self.assertEqual(
                result,
                {
                    "answer": "respuesta",
                    "citations": [101, 202],
                    "premise_status": status,
                },
            )
            self.assertEqual(payload, original)
            self.assertIsNot(result, payload)
            self.assertIsNot(result["citations"], payload["citations"])

    def test_invalid(self):
        good = {
            "answer": "respuesta",
            "citations": [101],
            "premise_status": "supported",
        }
        bad = [None, [], {}, dict(good, extra=True)]
        bad += [
            {key: value for key, value in good.items() if key != missing}
            for missing in good
        ]
        bad += [dict(good, answer=value) for value in [None, "", "  ", 123, []]]
        bad += [
            dict(good, citations=value)
            for value in [
                None,
                [],
                "101",
                [101, 101],
                [999],
                [True],
                [101.0],
                ["101"],
                [0],
                [-1],
                [101, 999],
                [{}],
                [[]],
                [None],
            ]
        ]
        bad += [
            dict(good, premise_status=value)
            for value in [None, "", "unknown", True, []]
        ]
        for payload in bad:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    validate_answer(payload, {101, 202})


if __name__ == "__main__":
    unittest.main()
