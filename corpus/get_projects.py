import requests
import os
import json
from dotenv import load_dotenv

# Carregar as variáveis de ambiente
load_dotenv()

API_KEY = os.getenv('LIBRARIES_IO_API_KEY')
BASE_URL = "https://libraries.io/api/search"
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_API_URL = "https://api.github.com/repos"

platforms = ["Maven", "NPM", "NuGet"]

def check_github_issues(repo_url):
    """Verifica se o repositório no GitHub possui issues habilitados."""
    try:
        owner, repo_name = repo_url.split('/')[-2], repo_url.split('/')[-1]
        github_url = f"{GITHUB_API_URL}/{owner}/{repo_name}"
        headers = {'Authorization': f'Bearer {GITHUB_TOKEN}'}
        response = requests.get(github_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('has_issues', False)
        else:
            print(f"Erro ao consultar GitHub API para {repo_url}: {response.status_code}")
            return False
    except Exception as e:
        print(f"Erro ao processar URL {repo_url}: {e}")
        return False

def fetch_projects_for_platform(platform):
    """Busca os projetos para uma plataforma específica."""
    params = {
        "platforms": platform,
        "sort": "rank",  
        "per_page": 30,  
        "page": 1,
        "api_key": API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Erro ao buscar projetos para a plataforma {platform}: {response.status_code}")
        return []

def save_projects_to_json(platform, projects):
    """Salva os projetos em um arquivo JSON no diretório correspondente."""
    dir_path = os.path.join('corpus', platform)
    os.makedirs(dir_path, exist_ok=True)  # Cria o diretório, se não existir
    
    file_path = os.path.join(dir_path, f"{platform}_projects.json")
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(projects, json_file, indent=4, ensure_ascii=False)

    print(f"{len(projects)} projetos salvos em: {file_path}")

def main():
    """Função principal para processar múltiplas plataformas."""
    for platform in platforms:
        print(f"Processando projetos para a plataforma: {platform}")
        projects = fetch_projects_for_platform(platform)
        valid_projects = []
        processed_repos = set()

        for project in projects:
            name = project.get("name", "N/A")
            stars = project.get("stars", 0)
            latest_release = project.get("latest_release_published_at", "N/A")
            repo_url = project.get("repository_url", "N/A")

            if stars < 10000:  # Ignorar projetos com menos de 10k estrelas
                continue
            
            if repo_url in processed_repos:  # Evitar duplicados
                continue
            processed_repos.add(repo_url)
            
            if repo_url and check_github_issues(repo_url):
                valid_projects.append({
                    'name': name,
                    'stars': stars,
                    'latest_release': latest_release,
                    'repo_url': repo_url
                })
        
        # Salvar os projetos válidos em um arquivo JSON
        save_projects_to_json(platform.lower(), valid_projects)

if __name__ == "__main__":
    main()
