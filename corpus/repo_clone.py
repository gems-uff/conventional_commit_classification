import os
import json
import subprocess

platforms = ["Maven", "NPM", "NuGet", "Pypi"]
output_base_dir = os.path.expanduser("~/Documents/repos")

def clone_repository(repo_url, repo_path, repo_name):
    """Clona um repositório no diretório especificado."""
    if os.path.exists(repo_path):
        print(f"Repositório {repo_name} já foi clonado. Pulando...")
        return "skipped"

    try:
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)
        print(f"Repositório {repo_name} clonado com sucesso!")
        return "cloned"
    except subprocess.CalledProcessError as e:
        print(f"Erro ao clonar {repo_name}: {e}")
        return "failed"

def process_platform(platform):
    """Processa uma plataforma específica, clonando os repositórios listados no JSON correspondente."""
    platform_dir = os.path.join(output_base_dir, platform.lower())
    os.makedirs(platform_dir, exist_ok=True)

    # Caminho do arquivo JSON para a plataforma
    json_file_path = os.path.join('corpus', platform.lower(), f"{platform.lower()}_projects.json")
    if not os.path.exists(json_file_path):
        print(f"Arquivo JSON não encontrado para a plataforma {platform}. Pulando...")
        return {"cloned": [], "skipped": [], "failed": []}

    # Ler os projetos do arquivo JSON
    with open(json_file_path, "r", encoding="utf-8") as file:
        projects = json.load(file)

    cloned = []
    skipped = []
    failed = []

    for project in projects:
        repo_url = project.get("repo_url")
        if not repo_url:
            print(f"Repositório sem URL válido encontrado. Pulando...")
            continue
        
        repo_name = repo_url.split('/')[-1]
        repo_path = os.path.join(platform_dir, repo_name)
        status = clone_repository(repo_url, repo_path, repo_name)

        # Adicionar o repositório à lista correspondente
        if status == "cloned":
            cloned.append(repo_name)
        elif status == "skipped":
            skipped.append(repo_name)
        elif status == "failed":
            failed.append(repo_name)

    return {"cloned": cloned, "skipped": skipped, "failed": failed}

def main():
    """Função principal para processar todas as plataformas e gerar relatório final."""
    overall_report = {"cloned": [], "skipped": [], "failed": []}

    for platform in platforms:
        print(f"\n--- Processando plataforma: {platform} ---")
        platform_report = process_platform(platform)

        overall_report["cloned"].extend(platform_report["cloned"])
        overall_report["skipped"].extend(platform_report["skipped"])
        overall_report["failed"].extend(platform_report["failed"])

    # Relatório final
    print("\n=== Relatório Final ===")
    print(f"Total de repositórios clonados: {len(overall_report['cloned'])}")
    print(f"Total de repositórios pulados: {len(overall_report['skipped'])}")
    print(f"Total de repositórios com falha: {len(overall_report['failed'])}")

    print("\nDetalhes:")
    print(f"Clonados: {overall_report['cloned']}")
    print(f"Pulados: {overall_report['skipped']}")
    print(f"Falhados: {overall_report['failed']}")

if __name__ == "__main__":
    main()
