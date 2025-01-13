import sys
import os
import json
import logging
import traceback
from multiprocessing import Pool
import re
import releasy
import requests
import datetime
from pathlib import Path

# Certifique-se de que REPO_PATH seja um objeto Path


# Configuração do logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adiciona o diretório acima ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util import (
    DATA_PATH,
    REPO_PATH,
    CPU,
    CORPUS_PATH
)

# Função para minerar os dados de um projeto
def mine_project_data(project_data: dict) -> dict:
    """
    Process a project's data to extract release-related metrics.

    Args:
        project_data (dict): A dictionary containing project information.

    Returns:
        dict: Processed project metrics including platform, releases, and timing details.
    """
    try:
        logger.info(f"Iniciando mineração para o projeto: {project_data.get('name')}")
        start_time = datetime.datetime.now()

        # Recupera o caminho do repositório e o nome do projeto
        platform = project_data['platform']
        repo_url = project_data['repo_url']
        try:
            owner, repo_name = repo_url.split("/")[-2], repo_url.split("/")[-1]
        except IndexError:
            logger.error(f"URL mal formatada: {repo_url}")
            return {}

        # O caminho do repositório de mineração
        repo_path = os.path.join(REPO_PATH, platform, repo_name)
        logger.debug(f"Repositório localizado em: {repo_path}")

        # Mineração dos dados do projeto
        project = releasy.Miner(repo_path, repo_name).apply(
            releasy.FinalReleaseMiner(),
            releasy.HistoryCommitMiner(),
            releasy.BaseReleaseMiner(),
            releasy.ContributorMiner(),
            releasy.SemanticReleaseMiner()
        ).mine()

        end_time = datetime.datetime.now()
        elapsed_seconds = (end_time - start_time).total_seconds()
        logger.info(f"Mineração concluída em {elapsed_seconds} segundos para o projeto {project_data.get('name')}")

        return {
            'platform': platform,
            'project': project.name,
            'owner': owner,  # Inclui o owner
            'prefixes': len(project.releases.prefixes()),
            'prefixes_names': " ".join(project.releases.prefixes()),
            'main_releases': len(project.main_releases),
            'patches': len(project.patches),
            'releases': len(project.releases),
            'time': elapsed_seconds,
        }


    except Exception as err:
        logger.error(f"Erro ao processar o projeto '{project_data['name']}': {err}")
        traceback.print_exc()
        return {}

# Função para contar prefixos incomuns
def uncommon_prefixes(qnt: int, names: str) -> int:
    """
    Count the number of uncommon prefixes in a list of release names.

    Args:
        qnt (int): Number of prefixes.
        names (str): Space-separated string of prefix names.

    Returns:
        int: Number of uncommon prefixes.
    """
    logger.debug(f"Contando prefixos incomuns para {qnt} prefixos: {names}")
    if qnt <= 1:
        return 0
    return len([name for name in names.split() if not re.match(r"^(|vv?|rel|l|rel/v|r|release)[/._-]?$", name, re.IGNORECASE)])

# Função para processar os projetos
def process_projects(projects_data):
    """
    Process a list of project data, separating selected and removed projects.

    Args:
        projects_data (list): List of dictionaries containing project information.

    Returns:
        tuple: Selected and removed projects as lists of dictionaries.
    """
    logger.info("Iniciando o processamento dos projetos")
    selected_projects = []
    removed_projects = []

    for project in projects_data:
        try:
            logger.debug(f"Processando projeto {project.get('name')}")
            if project.get('main_releases', 0) > 0:
                project['uncommon_prefixes'] = uncommon_prefixes(project['prefixes'], project['prefixes_names'])
                if project.get('uncommon_prefixes', 0) == 0:
                    selected_projects.append(project)
                else:
                    removed_projects.append(project)
            else:
                removed_projects.append(project)
        except Exception as e:
            logger.error(f"Erro ao processar o projeto {project.get('name')}: {e}")
            traceback.print_exc()

    logger.info("Processamento de projetos concluído")
    return selected_projects, removed_projects

# Função para salvar os resultados no JSON
def save_results_to_json(data, file_path):
    """
    Save results to a JSON file.

    Args:
        data (list): List of dictionaries to save.
        file_path (Path): Path to the output file.
    """
    logger.info(f"Salvando resultados em: {file_path}")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info("Resultados salvos com sucesso.")

# Função principal
if __name__ == "__main__":
    logger.info("Iniciando execução do script collect_corpus.py")

    # Lista de plataformas
    platforms = ['maven']  # Adicione mais plataformas conforme necessário
    all_projects_data = []

    # Carrega os projetos de cada plataforma
    for platform in platforms:
        platform_file = CORPUS_PATH / platform / f'{platform}_projects.json'

        if os.path.exists(platform_file):
            logger.info(f"Carregando dados da plataforma {platform}")
            with open(platform_file, 'r') as infile:
                platform_projects = json.load(infile)

                # Adiciona a plataforma ao projeto
                for project in platform_projects:
                    project['platform'] = platform
                    all_projects_data.append(project)
        else:
            logger.warning(f"A plataforma {platform} não tem projetos disponíveis.")

    results = []
    logger.info("Iniciando processamento paralelo dos projetos")
    # Processa os projetos com paralelismo
    with Pool(processes=CPU) as pool:
        for result in pool.imap_unordered(mine_project_data, all_projects_data):
            if result:
                results.append(result)

    # Salva os resultados completos em JSON no diretório DATA_PATH
    output_dir = DATA_PATH
    os.makedirs(output_dir, exist_ok=True)

    all_results_file = output_dir / f'{platform}_projects_all.json'
    save_results_to_json(results, all_results_file)

    # Separa projetos selecionados e removidos
    selected_projects, removed_projects = process_projects(results)

    # Salva os projetos selecionados e removidos em JSON
    selected_file = output_dir / f'{platform}_projects_selected.json'
    removed_file = output_dir / f'{platform}_projects_removed.json'
    save_results_to_json(selected_projects, selected_file)
    save_results_to_json(removed_projects, removed_file)

    logger.info(f"\nResultados salvos em {output_dir}")
