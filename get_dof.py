# encoding: utf-8

# Description: Script para descargar el Diario Oficial de la Federación (DOF) de México

# wget --https-only "https://diariooficial.gob.mx/abrirPDF.php?archivo=23012025-MAT.pdf&anio=2025&repo=repositorio/" -O "DOF_23012025-MAT.pdf

import os
import ssl
from datetime import datetime

import requests
import typer
from requests.adapters import HTTPAdapter


class TLSAdapter(HTTPAdapter):
    """Custom adapter to force TLS 1.2 or 1.3"""

    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context or ssl.create_default_context()
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self.ssl_context
        return super().init_poolmanager(*args, **kwargs)


# Create an SSL context with TLS 1.2 and reduced restrictions
ssl_context = ssl.create_default_context()
ssl_context.set_ciphers("DEFAULT@SECLEVEL=1")

session = requests.Session()
session.mount("https://", TLSAdapter(ssl_context=ssl_context))


def get_url(year, month, day):
    """Generate the URL for a given date"""
    base_url = "https://diariooficial.gob.mx/abrirPDF.php?archivo="
    return f"{base_url}{day}{month}{year}-MAT.pdf&anio={year}&repo=repositorio/"


def get_dof(year, month, day):
    """Download the DOF PDF for a given date"""
    url = get_url(year, month, day)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = session.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        return response.content
    else:
        return None  # Return None instead of raising an error


def is_file_valid(filepath: str) -> bool:
    """Check if a file exists and is a valid PDF"""
    if not os.path.exists(filepath):
        return False  # File does not exist

    if os.path.getsize(filepath) < 1000:  # Small files are likely invalid
        return False

    with open(filepath, "rb") as file:
        header = file.read(4)
        if not header.startswith(b"%PDF"):  # A valid PDF should start with %PDF
            return False

    return True  # The file is valid


def check_year_empty(year):
    """Check if ALL files for a given year are empty or invalid"""
    folder = f"./dof/{year}/"

    if not os.path.exists(folder):
        print(f"⚠️ The folder {folder} does not exist, assuming empty.")
        return True

    all_empty = True
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".pdf") and is_file_valid(os.path.join(root, file)):
                return False  # At least one valid file found, so the year is NOT empty

    print(f"🛑 ALL files for the year {year} are empty or missing.")
    return True


def save_dof(year, month, day):
    """Download and save the PDF only if it's not already valid."""
    folder = f"./dof/{year}/{month}"
    filename = f"{day}{month}{year}-MAT.pdf"
    filepath = os.path.join(folder, filename)

    # Skip if file already exists and is valid
    if is_file_valid(filepath):
        print(f"✅ File already exists and is valid: {filepath}. Skipping download.")
        return

    try:
        content = get_dof(year, month, day)

        # Validate the content
        if not content or len(content) < 1000 or not content.startswith(b"%PDF"):
            print(
                f"⚠️ File {filename} is empty, corrupt, or not a valid PDF. It will not be saved."
            )
            return

        os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist

        with open(filepath, "wb") as file:
            file.write(content)

        print(f"✅ File saved: {filepath}")

    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")


def main(
    start_year: int = typer.Option(datetime.now().year, "--start-year", help="Start year (defaults to current year)"),
    end_year: int = typer.Option(datetime.now().year - 50, "--end-year", help="End year (defaults to 50 years before current year)")
) -> None:
    """Download DOF PDFs for a range of years."""
    if start_year < end_year:
        typer.echo("Error: start-year must be greater than end-year (we count backwards)")
        raise typer.Exit(code=1)

    # Download PDFs
    month_days = {
        "01": 31,
        "02": 29,
        "03": 31,
        "04": 30,
        "05": 31,
        "06": 30,
        "07": 31,
        "08": 31,
        "09": 30,
        "10": 31,
        "11": 30,
        "12": 31,
    }

    _break = False
    empty_years_count = 0  # Count consecutive empty years

    for year in range(start_year, end_year, -1):
        typer.echo(f"\n🔎 Checking year {year}...\n")

        year_has_valid_files = False  # Reset flag for the current year

        for month, days in month_days.items():
            if _break:
                _break = False
                break
            for day in range(1, days + 1):
                # Stop if the day is greater than today (to prevent downloading future dates)
                _year = datetime.now().year
                _month = datetime.now().strftime("%m")
                _day = datetime.now().strftime("%d")
                if (
                    int(year) >= int(_year)
                    and int(month) >= int(_month)
                    and int(day) > int(_day)
                ):
                    _break = True
                    break

                save_dof(str(year), month, f"{day:02d}")  # f"{day:02d}" -> 01, 02, ...

                # If we find a valid file, mark the year as having data
                folder = f"./dof/{year}/{month}"
                filename = f"{day}{month}{year}-MAT.pdf"
                filepath = os.path.join(folder, filename)

                if is_file_valid(filepath):
                    year_has_valid_files = True

        # After processing each year, check if it was fully empty
        if check_year_empty(year):
            empty_years_count += 1
        else:
            empty_years_count = 0  # Reset count if we find a valid year

        # Stop downloading if we find 2 consecutive empty years
        if empty_years_count >= 2:
            typer.echo(f"🚨 Stopping process after {year} because the last 2 years were empty!")
            break


if __name__ == "__main__":
    typer.run(main)
