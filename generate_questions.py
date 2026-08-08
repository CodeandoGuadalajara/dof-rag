import os
import random
import typer
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
import google.generativeai as genai
import csv
import io

app = typer.Typer()
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def resolve_base_dir(arg: str) -> Path:
    """
    If the user passes "dof_markdown" and it exists in CWD, use it.
    Otherwise, look for the folder relative to the script's parent.
    """
    base = Path(arg)
    if base.exists():
        return base
    fallback = Path(__file__).resolve().parent.parent / arg
    if fallback.exists():
        return fallback
    typer.echo(f"❌ Directory not found: {arg}")
    raise typer.Exit()

def find_md_files(directory: Path) -> list[Path]:
    """
    Recursively traverses the folder, looking for .md files whose name
    is of the type DDMMYYYY- which indicates a DOF 2024 or 2025.
    """
    files = []
    for root, _, names in os.walk(directory):
        for name in names:
            if name.lower().endswith(".md"):
                stem = Path(name).stem
                # Format assumed: DDMMYYYY-... Ex.: 02042024-MAT
                if len(stem) >= 8 and stem[4:8] in ("2024", "2025"):
                    files.append(Path(root) / name)
    return files

def parse_csv_from_gemini(text: str) -> list[list[str]]:
    """
    Receives the text returned by Gemini and tries to parse a CSV,
    omitting triple backtick lines and the header row if included.
    Uses proper CSV parsing to handle commas within quoted fields.
    """
    lines = [line for line in text.splitlines() if not line.strip().startswith('```')]
    cleaned = '\n'.join(lines).strip()
    
    # Use the csv module's proper parsing capabilities
    reader = csv.reader(io.StringIO(cleaned), quotechar='"', doublequote=True)
    rows = []
    for row in reader:
        # Skip if row is empty or its first cell starts with "Question"
        if not row or (row[0].lower().startswith("question") or row[0].lower().startswith("pregunta")):
            continue
        rows.append(row)
    return rows

def generate_questions(file_path: Path, num: int) -> list[list[str]]:
    """
    Sends the prompt to Gemini to generate a CSV with
    EXACT columns: [Question,File,Page,Extract].
    Then parses and returns the list of rows (list[list[str]]).
    """
    content = file_path.read_text(encoding="utf-8")
    prompt = (
        f"Genera {num} preguntas simples basadas en este extracto del DOF (Diario Oficial de la Federación). "
        "Enfócate en secciones legales, resolutivas o normativas. "
        "Las preguntas deben ser claras, directas y respondibles a partir del texto. "
        "Las preguntas y respuestas DEBEN estar en ESPAÑOL. "
        "Devuelve un CSV con EXACTAMENTE estas columnas: Question,File,Page,Extract.\n\n"
        "- Question: Debe ser sencilla y enfocarse en hechos o información específica\n"
        "- File: Usa el nombre del archivo proporcionado abajo\n"
        "- Page: Indica el número de página donde se puede encontrar la respuesta\n"
        "- Extract: Incluye un extracto breve de 1-2 oraciones con la respuesta\n\n"
        "IMPORTANTE: Asegúrate de usar comillas dobles (\") para encerrar cualquier texto que contenga comas.\n\n"
        f"Nombre de archivo: {file_path.name}\n"
        f"---\n{content}\n---"
    )
    model = genai.GenerativeModel("gemini-2.0-flash")
    response_text = model.generate_content(prompt).text
    all_rows = parse_csv_from_gemini(response_text)
    
    # Verify we have at least one valid row
    if not all_rows:
        typer.echo("⚠️ No se generaron preguntas válidas, intentando nuevamente con un prompt simplificado...")
        # Fallback to a simpler prompt if the first one failed
        simple_prompt = (
            f"Genera {num} preguntas sencillas sobre hechos mencionados en este texto. "
            "Las preguntas y respuestas DEBEN estar en ESPAÑOL. "
            "Devuelve un CSV con columnas: Question,File,Page,Extract.\n\n"
            "IMPORTANTE: Usa comillas dobles (\") para cualquier texto que contenga comas.\n\n"
            f"Nombre de archivo: {file_path.name}\n"
            f"---\n{content}\n---"
        )
        response_text = model.generate_content(simple_prompt).text
        all_rows = parse_csv_from_gemini(response_text)
    
    # Force only 'num' rows, in case Gemini returns more
    return all_rows[:num]

@app.command()
def main(
    directory: str = typer.Argument(..., help="Path to dof_markdown"),
    output_csv: str = typer.Option("questions.csv", "--output", "-o"),
    num_questions: int = typer.Option(5, "--num", "-n")
):
    # 1) Resolve the base folder
    base_dir = resolve_base_dir(directory)
    # 2) Find .md files from 2024/2025
    md_files = find_md_files(base_dir)
    if not md_files:
        typer.echo("❌ No DOF 2024/2025 files found.")
        raise typer.Exit()

    # 3) Select a random file
    chosen = random.choice(md_files)
    typer.echo(f"📄 Selected: {chosen}")

    # 4) Generate the questions
    rows = generate_questions(chosen, num_questions)

    # 5) Write to CSV, in append mode, without duplicating headers
    run_id = datetime.now(timezone.utc).isoformat(timespec='seconds')
    file_path = Path(output_csv)
    file_exists = file_path.exists()
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Headers
        if not file_exists:
            writer.writerow(["RunTimestamp", "Question", "File", "Page", "Extract"])
        # Add rows
        for row in rows:
            # row is [Question,File,Page,Extract]
            writer.writerow([run_id] + row)

    typer.echo(f"✅ Added {len(rows)} questions to '{output_csv}' (RunID={run_id})")

if __name__ == "__main__":
    app()
