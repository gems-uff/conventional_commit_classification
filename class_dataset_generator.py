import json
from pathlib import Path

TARGET_LABEL = "feat" 

PLATFORM = "npm"

INPUT_FILE = Path("data") / "dataset" / f"formatted_{PLATFORM}.json"
OUTPUT_FILE = Path("data") / "dataset" / f"{TARGET_LABEL}_formatted_{PLATFORM}.json"

def main():
    try:
        with open(INPUT_FILE, "r") as infile:
            data = json.load(infile)
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo de entrada: {e}")
        return

    filtered_data = [entry for entry in data if entry["label"] == TARGET_LABEL]

    if not filtered_data:
        print(f"⚠️ Nenhum commit encontrado com a label '{TARGET_LABEL}'.")
        return

    try:
        with open(OUTPUT_FILE, "w") as outfile:
            json.dump(filtered_data, outfile, indent=4, ensure_ascii=False)
        print(f"✅ {len(filtered_data)} commits salvos em '{OUTPUT_FILE}'.")
    except Exception as e:
        print(f"❌ Erro ao salvar arquivo filtrado: {e}")

if __name__ == "__main__":
    main()
