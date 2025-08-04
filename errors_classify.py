import json
from pathlib import Path
from typing import List, Dict
from collections import Counter, defaultdict


class CommitResult:
    def __init__(self, data: Dict):
        self.input_text = data.get("input_text")
        self.type = data.get("type")  # predicted
        self.reason = data.get("reason")
        self.raw_response = data.get("raw_response")
        self.label = data.get("label")  # ground truth
        self.correct = data.get("correct", False)
        self.breaking_change = data.get("breaking_change", False)
        self.expected_breaking_change = data.get("expected_breaking_change", False)
        self.breaking_correct = data.get("breaking_correct", False)
        self.processing_time = data.get("processing_time", 0.0)
        self.commit_hash = data.get("commit_hash")
        self.timestamp = data.get("timestamp")

    def to_dict(self):
        return {
            "input_text": self.input_text,
            "type": self.type,
            "label": self.label,
            "reason": self.reason,
            "raw_response": self.raw_response,
            "breaking_change": self.breaking_change,
            "breaking_correct": self.breaking_correct,
            "commit_hash": self.commit_hash,
        }


def analyze_all_labels(model: str, platform: str, extension: str, strategy: str):
    input_path = Path(f"data/output/{model}/{platform}/{strategy}_{extension}.json")
    output_dir = Path(f"data/error/{model}/{platform}/{strategy}_{extension}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_labels.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: List[Dict] = data.get("results", [])
    all_labels = set(r.get("label") for r in results if r.get("label"))

    output_data = {}

    for target_label in all_labels:
        false_negatives = []
        false_positives = []
        fn_counter = Counter()
        fp_counter = Counter()

        for item in results:
            r = CommitResult(item)

            if r.label == target_label and not r.correct:
                false_negatives.append(r.to_dict())
                fn_counter[r.type] += 1  # modelo previu outro tipo

            elif r.type == target_label and not r.correct:
                false_positives.append(r.to_dict())
                fp_counter[r.label] += 1  # modelo disse que era target_label, mas era outro

        output_data[target_label] = {
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "false_negatives_summary": dict(fn_counter),
            "false_positives_summary": dict(fp_counter),
        }

        print(f"Label '{target_label}': {len(false_negatives)} FN, {len(false_positives)} FP")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n🔍 Resultado salvo em: {output_path}")


if __name__ == "__main__":
    model = "qwen2.5-coder:32b"
    platform = "npm"
    extension = "v7"
    strategy = "E2"
    
    analyze_all_labels(model, platform, extension, strategy)
