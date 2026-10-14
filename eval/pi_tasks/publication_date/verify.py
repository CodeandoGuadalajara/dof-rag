import unittest

from dates import publication_date


class PublicationTests(unittest.TestCase):
    def test_dates(self):
        months = [
            "enero",
            "febrero",
            "marzo",
            "abril",
            "mayo",
            "junio",
            "julio",
            "agosto",
            "septiembre",
            "octubre",
            "noviembre",
            "diciembre",
        ]
        for month_number, month in enumerate(months, 1):
            with self.subTest(month=month):
                self.assertEqual(
                    publication_date(f"Publicado el 4 de {month} de 2016."),
                    f"2016-{month_number:02d}-04",
                )
        self.assertEqual(
            publication_date("FECHA: 1  DE  SETIEMBRE DE 2026."), "2026-09-01"
        )
        self.assertEqual(publication_date("29 de febrero de 2000"), "2000-02-29")
        self.assertEqual(
            publication_date("Fecha: 9\u00a0de\u00a0agosto de 2006."), "2006-08-09"
        )
        self.assertEqual(
            publication_date("4 de febrero de 2016 y 5 de marzo de 2017"), "2016-02-04"
        )

    def test_invalid(self):
        for text in [
            "sin fecha",
            "30 de febrero de 2024",
            "29 de febrero de 1900",
            "29 de febrero de 2023",
            "1 de inventado de 2026",
            "0 de enero de 2026",
            "32 de enero de 2026",
            "31 de abril de 2026",
            "30 de febrero de 2024 y 4 de marzo de 2024",
        ]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    publication_date(text)


if __name__ == "__main__":
    unittest.main()
