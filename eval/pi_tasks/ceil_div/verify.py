import unittest

from calc import ceil_div


class CeilingTests(unittest.TestCase):
    def test_signed_and_large_values(self):
        cases = [(a, b) for a in range(-10, 11) for b in range(-5, 6) if b]
        cases += [(10**50 + 1, 3), (-(10**50 + 1), 3), (10**50 + 1, -3), (0, -3)]
        for a, b in cases:
            with self.subTest(a=a, b=b):
                actual = ceil_div(a, b)
                self.assertIs(type(actual), int)
                self.assertEqual(actual, -(a // -b))

    def test_zero(self):
        with self.assertRaises(ZeroDivisionError):
            ceil_div(1, 0)


if __name__ == "__main__":
    unittest.main()
