## 2 - Preparação dos Commits ##
### Imports e configuração ###
import json
import logging
import subprocess
import traceback
import sys
import os
import re
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Carregar variaveis de ambiente
load_dotenv()

REPO_PATH = Path("/home/rodrigoscarvalho/repos")
DATA_PATH = Path("data")

CONVENTIONAL_COMMITS_PATTERN = re.compile(r"^[a-z]+(\([a-zA-Z]+\))?: .*")
CUTOFF_DATE = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
FILTER_BY_DATE = True
FILTER_BY_CONVENTION = False  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

### Funções de Commit ###
def get_git_show(repo_path, commit_hash):
    try:
        result = subprocess.run(
            ['git', 'show', commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode == 0:
            return result.stdout
        else:
            logging.error(f"Error running git show for commit {commit_hash}: {result.stderr}")
            return None
    except Exception as e:
        logging.error(f"Error executing git show command for commit {commit_hash}: {str(e)}")
        return None

def get_commit_message(repo_path, commit_hash):
    try:
        result = subprocess.run(
            ['git', 'show', '-s', '--format=%B', commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception as e:
        logging.error(f"Error getting commit message for {commit_hash}: {str(e)}")
        return None

def get_commit_date(repo_path, commit_hash):
    try:
        result = subprocess.run(
            ['git', 'show', '-s', '--format=%cI', commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode == 0:
            date_str = result.stdout.strip()
            return datetime.datetime.fromisoformat(date_str)
        return None
    except Exception as e:
        logging.error(f"Error getting commit date for {commit_hash}: {str(e)}")
        return None

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

### Função principal ###
def mine(project: dict) -> dict:
    project_data = {}
    logging.info(f"📌 Processing project: {project['project']} ({project['platform']})")

    try:
        platform = project['platform']
        repo_name = project['project']
        repo_path = str(REPO_PATH / platform / repo_name)

        logging.info(f"📂 Repository: {repo_path}")
        
        platform_dir = DATA_PATH / "conventional_commits" / platform
        platform_dir.mkdir(parents=True, exist_ok=True)

        combined_data = {
            'platform': platform,
            'project': repo_name,
            'commits': []
        }

        result = subprocess.run(
            ['git', 'log', '--format=%H'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        if result.returncode != 0:
            raise Exception(f"Failed to get commits: {result.stderr}")

        commits = result.stdout.strip().split('\n')
        logging.info(f"Found {len(commits)} commits in {repo_name}")

        commits_processed = 0
        commits_filtered_by_date = 0
        commits_filtered_by_convention = 0

        for commit_hash in commits:
            commits_processed += 1

            if FILTER_BY_DATE:
                commit_date = get_commit_date(repo_path, commit_hash)
                if commit_date and commit_date < CUTOFF_DATE:
                    commits_filtered_by_date += 1
                    continue

            commit_message = get_commit_message(repo_path, commit_hash)
            if not commit_message:
                continue

            if FILTER_BY_CONVENTION and not CONVENTIONAL_COMMITS_PATTERN.match(commit_message):
                commits_filtered_by_convention += 1
                continue

            git_show = get_git_show(repo_path, commit_hash)

            if git_show:
                release_name = get_commit_release(repo_path, commit_hash) 
                commit_data = {
                    'message': commit_message,
                    'git_show': git_show,
                    'release': release_name
                }
                combined_data['commits'].append(commit_data)

        output_file =  platform_dir / f"{repo_name}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=4, ensure_ascii=False)

        logging.info(f"📁 File saved: {output_file.resolve()}")
        logging.info(f"📊 Statistics:")
        logging.info(f"   - Total commits processed: {commits_processed}")
        logging.info(f"   - Commits filtered by date: {commits_filtered_by_date}")
        logging.info(f"   - Commits filtered by convention: {commits_filtered_by_convention}")
        logging.info(f"   - Commits included in output: {len(combined_data['commits'])}")

    except Exception as err:
        logging.error(f"❌ Error processing {platform}/{repo_name}: {err}")
        traceback.print_exc()

    return project_data

### Excutar ###
logging.info("🚀 Starting project processing!")
if FILTER_BY_DATE:
    logging.info(f"📅 Date filtering enabled - processing only commits after {CUTOFF_DATE.isoformat()}")

platforms = ['npm']
results = []

for platform in platforms:
    input_file = Path(DATA_PATH) / "projects" / f"{platform}_projects_selected.json"
    logging.info(f"📄 Loading projects from {input_file}")
    
    with open(input_file, mode='r') as file:
        projects = json.load(file)

    for project in projects:
        if project.get("conventional_commits_ratio", 0) < 0.1:
            logging.info(f"🔍 Skipping {project['project']} due to low conventional commits ratio ({project['conventional_commits_ratio']})")
            continue
        result = mine(project)
        results.append(result)

logging.info("🏁 Processing completed!")
