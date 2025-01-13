import requests
import json
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import DATA_PATH
DATA_PATH = Path(DATA_PATH)

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GRAPHQL_URL = "https://api.github.com/graphql"

query = """query ($owner: String!, $repo: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    issues(first: 100, after: $cursor, states: [OPEN, CLOSED]) {
      nodes {
        number
        title
        body
        url
        labels(first: 10) {
          nodes {
            name
          }
        }
        milestones(first: 1) {
          nodes {
            title
          }
        }
        comments(first: 20) {
          nodes {
            author {
              login
            }
            body
          }
        }
        assignees(first: 10) {
          nodes {
            login
          }
        }
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}

"""

path = os.getcwd()
headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}

def fetch_all_issues(owner, repo):
    issues = []
    cursor = None
    while True:
        variables = {"owner": owner, "repo": repo, "cursor": cursor}
        response = requests.post(
            GRAPHQL_URL, json={"query": query, "variables": variables}, headers=headers
        )
        if response.status_code != 200:
            print(f"Erro ao buscar issues para {owner}/{repo}: {response.status_code}")
            return []
        data = response.json()
        if "errors" in data:
            print(f"Erro retornado pela API: {data['errors']}")
            return []
        issues.extend(data["data"]["repository"]["issues"]["nodes"])
        page_info = data["data"]["repository"]["issues"]["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]
    return issues

def save_issues_to_file(issues, platform, repo):
    output_dir = os.path.join(path, "issues", platform, repo)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{repo}.json")
    with open(output_file, "w") as f:
        json.dump(issues, f, indent=4)
    print(f"Issues salvas em: {output_file}")

def process_repositories(platform, projects):
    for project in projects:
        repo_name = project["project"]
        repo_platform = project["platform"]
        if repo_platform != platform:
            continue

        print(f"Processando {repo_platform}/{repo_name}...")

        issues = fetch_all_issues(project["owner"], repo_name)
        if issues:
            save_issues_to_file(issues, platform, repo_name)
        else:
            print(f"Não foi possível obter issues para {repo_platform}/{repo_name}.")

def process_platforms(platforms):
    for platform in platforms:
        input_file = os.path.join(DATA_PATH, f"{platform}_projects_selected.json")
        
        if not os.path.exists(input_file):
            print(f"Arquivo de projetos para a plataforma {platform} não encontrado.")
            continue
        
        with open(input_file, mode='r') as file:
            projects = json.load(file)
        
        process_repositories(platform, projects)

# Lista de plataformas a serem processadas
platforms = ['maven']  # Adicione outras plataformas conforme necessário
process_platforms(platforms)
