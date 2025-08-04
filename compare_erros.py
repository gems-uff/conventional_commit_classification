import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set


def load_errors_by_hash_from_label(data: Dict, error_type: str) -> Dict[str, Dict]:
    """
    Recebe dados de um label específico, extrai os erros do tipo ('false_positives' ou 'false_negatives'),
    e retorna um dicionário com os commit_hash como chave.
    """
    return {
        item["commit_hash"]: item
        for item in data.get(error_type, [])
        if item.get("commit_hash")
    }


def count_by_type(errors: List[Dict]) -> Dict[str, int]:
    """
    Conta quantos erros existem por tipo (por exemplo: feat, fix, refactor).
    """
    return dict(Counter(error.get("type", "unknown") for error in errors))


def count_by_label(errors: List[Dict]) -> Dict[str, int]:
    """
    Conta quantos erros existem por label (o rótulo correto), útil para false_positives.
    """
    return dict(Counter(error.get("label", "unknown") for error in errors))

def group_by_label(errors: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = {}
    for err in errors:
        label = err.get("label", "unknown")
        grouped.setdefault(label, []).append(err)
    return grouped

def group_by_type(errors: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = {}
    for err in errors:
        type_ = err.get("type", "unknown")
        grouped.setdefault(type_, []).append(err)
    return grouped


def compare_errors_all_labels(
    model: str,
    platform: str,
    extension_v1: str,
    extension_v2: str,
    base_path="data/error"
):
    file_v1 = Path(base_path) / model / platform / extension_v1 / "all_labels.json"
    file_v2 = Path(base_path) / model / platform / extension_v2 / "all_labels.json"

    output_dir = Path(base_path) / model / platform
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"compare_{extension_v1}_vs_{extension_v2}_all_labels.json"

    with open(file_v1, "r", encoding="utf-8") as f1, open(file_v2, "r", encoding="utf-8") as f2:
        data_v1 = json.load(f1)
        data_v2 = json.load(f2)

    all_labels: Set[str] = set(data_v1.keys()) | set(data_v2.keys())
    all_output = {}

    for label in sorted(all_labels):
        label_v1 = data_v1.get(label, {})
        label_v2 = data_v2.get(label, {})

        label_output = {}

        for error_type in ["false_positives", "false_negatives"]:
            v1_errors = load_errors_by_hash_from_label(label_v1, error_type)
            v2_errors = load_errors_by_hash_from_label(label_v2, error_type)
        
            set_v1 = set(v1_errors)
            set_v2 = set(v2_errors)
        
            if error_type == "false_positives":
                corrected_hashes = set_v1 - set_v2
                corrected = [v1_errors[h] for h in corrected_hashes]
                maintained = [v1_errors[h] for h in set_v1 & set_v2]
                new = [v2_errors[h] for h in set_v2 - set_v1]
            else:  # false_negatives
                corrected_hashes = set_v1 - set_v2
                corrected = [v1_errors[h] for h in corrected_hashes]
                maintained = [v1_errors[h] for h in set_v1 & set_v2]
                new = [v2_errors[h] for h in set_v2 - set_v1]


            summary_fn = count_by_label if error_type == "false_positives" else count_by_type
            group_fn = group_by_label if error_type == "false_positives" else group_by_type


            label_output[error_type] = {
                "v1": list(v1_errors.values()),
                "v2": list(v2_errors.values()),
                "corrected": group_fn(corrected),
                "maintained": group_fn(maintained),
                "new": group_fn(new),
                "summary_by_type": {
                    "corrected": summary_fn(corrected),
                    "maintained": summary_fn(maintained),
                    "new": summary_fn(new),
                }
            }




            print(f"\n[{label}] {error_type.replace('_', ' ').title()}:")
            print(f"  v1: {len(v1_errors)} erros")
            print(f"  v2: {len(v2_errors)} erros")
            print(f"  Corrigidos: {len(corrected)} {summary_fn(corrected)}")
            print(f"  Mantidos:   {len(maintained)} {summary_fn(maintained)}")
            print(f"  Novos:      {len(new)} {summary_fn(new)}")


        all_output[label] = label_output

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_output, f, indent=2, ensure_ascii=False)

    print(f"\n📦 Resultado completo salvo em {output_file}")


if __name__ == "__main__":
    model = "qwen2.5-coder:32b"
    platform = "npm"
    extension_v1 = "E2_v6"
    extension_v2 = "E2_v7"
    
    compare_errors_all_labels(model, platform, extension_v1, extension_v2)
