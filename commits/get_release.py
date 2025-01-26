import datetime
import json
import pygit2
import releasy
import traceback
from multiprocessing import Pool
import sys
import os
import re
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import DATA_PATH, REPO_PATH, ISSUE_PATH, CPU, delta2days, get_cycle, get_duration, get_start_delay, get_commits_in_advance

DATA_PATH = Path(DATA_PATH)
REPO_PATH = Path(REPO_PATH)
ISSUE_PATH = Path(ISSUE_PATH)

def process_issues_association(project: dict) -> None:
    """
    Realiza a associação entre commits e issues e consolida as issues no nível de release.
    """
    platform = project['platform']
    repo_name = project['project']

    project_dir = DATA_PATH / platform / repo_name
    issues_file = ISSUE_PATH / platform / repo_name / f"{repo_name}.json"

    if issues_file.exists():
        with open(issues_file, 'r', encoding='utf-8') as f:
            issues_data = json.load(f)
    else:
        print(f"Arquivo de issues não encontrado para {repo_name}. Pulando associação.")
        return

    issues_map = {str(issue['number']): issue['title'] for issue in issues_data}

    for output_file in project_dir.glob(f"{repo_name}_*.json"):
        with open(output_file, 'r', encoding='utf-8') as f:
            release_data = json.load(f)

        all_issues = {}  # Dicionário consolidado para armazenar todas as issues da release

        for commit in release_data.get('commits_info', []):
            commit_message = commit['message']
            # Buscar referências a issues no formato #número ou gh-número
            referenced_issues = re.findall(r'#(\d+)|gh-(\d+)', commit_message)

            for match in referenced_issues:
                issue_number = match[0] or match[1]  # Pega o número detectado
                if issue_number in issues_map:
                    # Adiciona ao dicionário consolidado (garantindo que não haja duplicatas)
                    if issue_number not in all_issues:
                        all_issues[issue_number] = {
                            'number': issue_number,
                            'title': issues_map[issue_number]
                        }

        # Adicionar a lista consolidada de issues no nível da release
        release_data['issues'] = list(all_issues.values())

        # Remover as issues de cada commit
        for commit in release_data.get('commits_info', []):
            commit.pop('issues', None)

        # Salva novamente o arquivo com as associações
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(release_data, f, indent=4, ensure_ascii=False)

def mine(project: dict) -> tuple:
    elapsed_time = datetime.datetime.now()
    project_data = {}

    try:
        platform = project['platform']
        repo_name = project['project']
        repo_path = str(REPO_PATH / platform / repo_name)

        # Criar estrutura de diretórios
        platform_dir = DATA_PATH / platform
        project_dir = platform_dir / repo_name
        project_dir.mkdir(parents=True, exist_ok=True)

        project = releasy.Miner(repo_path, repo_name).apply(
            releasy.FinalReleaseMiner(),
            releasy.HistoryCommitMiner(),
            releasy.BaseReleaseMiner(),
            releasy.ContributorMiner(),
            releasy.SemanticReleaseMiner()
        ).mine()

        all_releases = project.main_releases
        all_patches = project.patches  # Patches adicionais
        repo = pygit2.Repository(repo_path)

        # Processando as main releases
        for release in all_releases:
            if not release.commits or not release.prev_main_release:
                continue

            cycle_days = delta2days(get_cycle(release))
            duration = delta2days(get_duration(release))
            start_delay = delta2days(get_start_delay(release))
            commits_in_advance = get_commits_in_advance(release)

            commits_info = []
            for commit in release.commits:
                commit_obj = repo.get(commit.id)
                commit_message = commit_obj.message.strip()

                commits_info.append({
                    'hash': commit.id,
                    'message': commit_message,
                    'issues': []  # Campo vazio, preenchido posteriormente
                })

            release_data = {
                'platform': platform,
                'project': repo_name,
                'release': release.name,
                'previous_release': release.prev_main_release.name if release.prev_main_release else '',
                'cycle_days': cycle_days,
                'commits': len(release.commits),
                'commits_in_advance': len(commits_in_advance),
                'start_delay': start_delay,
                'duration': duration,
                'commits_info': commits_info
            }

            # Salvar arquivo da main release
            output_file = project_dir / f"{repo_name}_{release.name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(release_data, f, indent=4, ensure_ascii=False)

        # Processando os patches como releases independentes
        for patch in all_patches:
            if not patch.commits:
                continue

            cycle_days = delta2days(get_cycle(patch))
            duration = delta2days(get_duration(patch))
            start_delay = delta2days(get_start_delay(patch))
            commits_in_advance = get_commits_in_advance(patch)

            commits_info = []
            for commit in patch.commits:
                commit_obj = repo.get(commit.id)
                commit_message = commit_obj.message.strip()

                commits_info.append({
                    'hash': commit.id,
                    'message': commit_message,
                    'issues': []  # Campo vazio, preenchido posteriormente
                })

            patch_data = {
                'platform': platform,
                'project': repo_name,
                'release': patch.name,
                'previous_release': patch.prev_main_release.name if patch.prev_main_release else '',
                'cycle_days': cycle_days,
                'commits': len(patch.commits),
                'start_delay': start_delay,
                'duration': duration,
                'commits_info': commits_info
            }

            # Salvar arquivo do patch
            output_file = project_dir / f"{repo_name}_{patch.name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(patch_data, f, indent=4, ensure_ascii=False)

        elapsed_time = datetime.datetime.now() - elapsed_time
        project_data = {
            'project': repo_name,
            'time': elapsed_time.total_seconds()
        }

    except Exception as err:
        print(f"{platform}/{repo_name:40} {err=}")
        traceback.print_exception(err)

    return project_data

# Processamento por plataforma
platforms = ['maven']  # Adicione as plataformas necessárias
results = []

for platform in platforms:
    input_file = DATA_PATH / f"{platform}_projects_selected.json"

    with open(input_file, mode='r') as file:
        projects = json.load(file)

    with Pool(processes=CPU) as pool:
        platform_results = pool.map(mine, projects)
        results.extend(platform_results)

# Após processar todos os projetos, realizar associação com issues
for platform in platforms:
    input_file = DATA_PATH / f"{platform}_projects_selected.json"

    with open(input_file, mode='r') as file:
        projects = json.load(file)

    for project in projects:
        process_issues_association(project)