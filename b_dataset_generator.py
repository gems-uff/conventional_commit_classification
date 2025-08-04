import json
import re
import sys
import os
import logging
import subprocess
from pathlib import Path
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dataset_creator.log"),
        logging.StreamHandler(sys.stdout),
    ],
)


load_dotenv()

REPO_PATH = Path("/home/rodrigoscarvalho/repos")
DATA_PATH = Path("data")


VALID_TYPES = {"fix", "feat", "build", "chore", "ci", "docs", "style", "refactor", "perf", "test"}
CONVENTIONAL_COMMITS_PATTERN = re.compile(r"^([a-z]+)(\([^)]+\))?:\s*")
BREAKING_CHANGE_FOOTER = re.compile(r"breaking change:", re.IGNORECASE)

def format_git_show(raw_git_show: str, commit_message: str) -> str:
    lines = raw_git_show.splitlines()
    formatted_lines = []

    commit_msg_lines = commit_message.strip().splitlines()
    formatted_lines.append("Commit Message:")
    formatted_lines.extend(commit_msg_lines)
    formatted_lines.append("\nCode Changes:")

    current_file = None
    file_changes = []

    def flush_file_changes():
        if current_file and file_changes:
            formatted_lines.append(f"\nFile: {current_file}")
            formatted_lines.extend(file_changes)

    for line in lines:
        if line.startswith("diff --git"):
            flush_file_changes()
            file_changes = []
            parts = line.split()
            if len(parts) >= 4:
                a_path = parts[2][2:]  # remove 'a/'
                current_file = a_path
            else:
                current_file = None
        elif line.startswith('+++') or line.startswith('---'):
            continue
        elif line.startswith("@@"):
            continue
        elif line.startswith('+') and not line.startswith('+++'):
            file_changes.append(f"+ {line[1:]}")
        elif line.startswith('-') and not line.startswith('---'):
            file_changes.append(f"- {line[1:]}")
    flush_file_changes()

    return "\n".join(formatted_lines)


def classify_commit(message):
    message_lower = message.lower()
    first_line = message.split("\n", 1)[0].strip()

    
    is_breaking = False
    if re.match(r"^[a-z]+(\([^)]+\))?!:", first_line):
        is_breaking = True
    elif "breaking change:" in message_lower:
        is_breaking = True

    
    conventional_types = [
        "fix", "feat", "build", "chore", "ci", "docs",
        "style", "refactor", "perf", "test"
    ]

    type_match = re.match(r"^([a-z]+)(\([^)]+\))?:", first_line)
    if type_match:
        commit_type = type_match.group(1).lower()
        if commit_type in conventional_types:
            return commit_type, is_breaking

    
    return "others", is_breaking

def remove_prefix(message):
    return re.sub(CONVENTIONAL_COMMITS_PATTERN, "", message, count=1).strip()

def clean_git_show(git_show, message):
    
    cleaned_message = remove_prefix(message)

    
    cleaned_message = re.sub(r"(?i)breaking change:", "", cleaned_message)

    message_start = git_show.find(message.split("\n")[0])

    if message_start != -1:
        before = git_show[:message_start]
        after = git_show[message_start + len(message):]
        lines = git_show[message_start:message_start + len(message)].split("\n")

        if lines:
            indent = len(lines[0]) - len(lines[0].lstrip())
            cleaned_lines = [" " * indent + line for line in cleaned_message.split("\n")]
            cleaned_with_indent = "\n".join(cleaned_lines)
            return before + cleaned_with_indent + after

    return git_show


def get_commit_release(repo_path, commit_hash):
    try:
        result = subprocess.run(
            ['git', 'tag', '--contains', commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode == 0:
            tags = result.stdout.strip().split('\n')
            return tags[0] if tags else "unknown"
        else:
            logging.warning(f"Warning getting release for {commit_hash}: {result.stderr}")
            return "unknown"
    except Exception as e:
        logging.error(f"Error getting release for {commit_hash}: {str(e)}")
        return "unknown"


platforms = ["npm"]

for platform in platforms:
    logging.info(f"Processando a plataforma: {platform}")

    output_file = DATA_PATH / "dataset" / f"formatted_{platform}.json"
    formatted_data = []
    seen_commits = set()

    label_counts = {}

    input_file = DATA_PATH / "conventional_commits" / platform / "*.json"

    for json_file in Path(input_file).parent.glob(Path(input_file).name):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            logging.error(f"Erro ao ler {json_file}: {e}")
            continue

        repo_path = REPO_PATH / platform / data.get("project", "")

        for commit in data.get("commits", []):
            message = commit["message"]
            git_show = commit["git_show"]
            cleaned_git_show = format_git_show(git_show, message)
            label, breaking_change = classify_commit(message)
            release = get_commit_release(repo_path, commit_hash=commit.get("hash", "")) if "hash" in commit else "unknown"

            label_counts[label] = label_counts.get(label, 0) + 1

            commit_identifier = (cleaned_git_show, label)

            if commit_identifier in seen_commits:
                continue

            seen_commits.add(commit_identifier)
            formatted_data.append({
                "input_text": cleaned_git_show,
                "label": label,
                "message": message,
                "breaking_change": breaking_change,
                "release": commit.get("release", "unknown")
            })


    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(formatted_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Dataset criado para '{platform}' e salvo em '{output_file}'.")
        logging.info(f"📊 Quantidade de labels para {platform}:")
        for label, count in label_counts.items():
            logging.info(f"   - {label}: {count}")
    except Exception as e:
        logging.error(f"Falha ao salvar dataset para {platform}: {e}")
