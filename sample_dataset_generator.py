import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from dotenv import load_dotenv
from scipy.stats import norm
from sklearn.model_selection import train_test_split

# --- Configuração de Logging e Constantes ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dataset_creator.log", mode='w'),
        logging.StreamHandler(sys.stdout),
    ],
)

load_dotenv()

REPO_PATH = Path(os.getenv("REPO_PATH", "/home/rodrigoscarvalho/repos"))
DATA_PATH = Path("data")
VALID_TYPES = {"fix", "feat", "build", "chore", "ci", "docs", "style", "refactor", "perf", "test"}
CONVENTIONAL_COMMITS_PATTERN = re.compile(r"^[a-z]+(\([^)]+\))?!?:")


def extract_commit_hash_from_git_show(git_show: str) -> str:
    """
    Extrai o hash do commit da saída do git show.
    """
    if not git_show:
        return "unknown"
        
    for line in git_show.splitlines():
        if line.startswith("commit "):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return "unknown"


def format_git_show(raw_git_show: str, cleaned_commit_message: str, max_lines_per_file: int = 300) -> str:
    """
    Formata a saída do 'git show' em uma única passagem, criando um texto limpo
    e legível para o LLM, incluindo contexto e lógica de truncamento.
    """
    if not raw_git_show or not raw_git_show.strip():
        return f"Commit Message:\n{cleaned_commit_message}\n\nCode Changes:\n(no diff available)"
    
    lines = raw_git_show.splitlines()
    output_lines: List[str] = []
    
    # --- Processa Metadados e Diff em uma única passagem ---
    git_metadata: Dict[str, str] = {}
    
    # Separa o 'git show' em blocos de diff
    raw_diff_parts = raw_git_show.split('diff --git')
    header_and_message = raw_diff_parts[0] if raw_diff_parts else ""

    # Processa o cabeçalho
    for line in header_and_message.splitlines():
        if line.startswith("commit "):
            git_metadata['hash'] = line.split()[1] if len(line.split()) > 1 else 'unknown'
        elif line.startswith("Author:"):
            git_metadata['author'] = line.replace("Author:", "").strip()
        elif line.startswith("Date:"):
            git_metadata['date'] = line.replace("Date:", "").strip()
    
    output_lines.append(f"Commit Hash: {git_metadata.get('hash', 'unknown')}")
    output_lines.append(f"Author: {git_metadata.get('author', 'unknown')}")
    output_lines.append(f"Date: {git_metadata.get('date', 'unknown')}\n")

    # Adiciona a mensagem de commit limpa
    output_lines.append("Commit Message:")
    message_lines = cleaned_commit_message.strip().splitlines()
    output_lines.extend(message_lines if message_lines else ["(no message)"])
    output_lines.append("\nCode Changes:")

    # Processa cada bloco de diff
    if len(raw_diff_parts) > 1:
        for diff_part in raw_diff_parts[1:]:
            if not diff_part.strip():
                continue
                
            full_diff_block = 'diff --git' + diff_part
            diff_lines = full_diff_block.splitlines()
            
            # Extrai o nome do arquivo de forma mais robusta
            try:
                first_line = diff_lines[0] if diff_lines else ""
                if ' a/' in first_line and ' b/' in first_line:
                    file_path = first_line.split(' a/')[1].split(' b/')[0]
                else:
                    file_path = "unknown_file"
            except (IndexError, AttributeError):
                file_path = "unknown_file"
            
            output_lines.append(f"\n--- File: {file_path} ---")
            
            # Aplica o truncamento
            limited_diff = diff_lines[:max_lines_per_file]
            output_lines.extend(limited_diff)
            if len(diff_lines) > max_lines_per_file:
                output_lines.append(f"... [diff truncated, showing first {max_lines_per_file} lines] ...")
    else:
        output_lines.append("(no code changes detected)")

    return "\n".join(output_lines)


def calculate_sample_size(population_size: int, margin_error: float = 0.05, confidence_level: float = 0.95) -> int:
    """
    Calcula o tamanho da amostra usando a fórmula de correção para populações finitas.
    """
    if population_size == 0:
        return 0
    
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    p = 0.5  # Proporção mais conservadora
    n_infinite = (z**2 * p * (1 - p)) / (margin_error**2)
    n = n_infinite / (1 + (n_infinite - 1) / population_size)
    return int(round(n))


def classify_commit(message: str) -> Tuple[str, bool]:
    """
    Classifica o commit baseado na mensagem e detecta breaking changes.
    """
    if not message or not message.strip():
        return "others", False
        
    first_line = message.strip().split('\n')[0]
    
    is_breaking = False
    try:
        match = CONVENTIONAL_COMMITS_PATTERN.match(first_line)
        if match:
            matched_text = match.group(0)
            is_breaking = '!' in matched_text
        
        # Verifica também por BREAKING CHANGE no corpo da mensagem
        is_breaking = is_breaking or "BREAKING CHANGE:" in message.upper()
    except Exception as e:
        logging.warning(f"Erro ao verificar breaking change: {e}")
    
    # Classifica o tipo do commit
    try:
        match = CONVENTIONAL_COMMITS_PATTERN.match(first_line)
        if match:
            commit_type = match.group(0).split('(')[0].split('!')[0].replace(':', '')
            if commit_type in VALID_TYPES:
                return commit_type, is_breaking
    except Exception as e:
        logging.warning(f"Erro ao classificar commit: {e}")
    
    return "others", is_breaking


def remove_prefix(message: str) -> str:
    """
    Remove o prefixo do conventional commit da mensagem.
    """
    if not message:
        return ""
    return CONVENTIONAL_COMMITS_PATTERN.sub("", message, count=1).strip()


def process_platform(platform: str, margin_error: float, confidence_level: float):
    """Função principal para processar todos os commits de uma plataforma."""
    logging.info(f"🚀 Processando a plataforma: {platform}")

    all_commits_data = []
    seen_commits = set()  # Para evitar duplicatas
    json_files_path = DATA_PATH / "conventional_commits" / platform
    
    # Verifica se o diretório existe
    if not json_files_path.exists():
        logging.error(f"Diretório não encontrado: {json_files_path}")
        logging.info(f"Estrutura de diretórios esperada: {DATA_PATH}/conventional_commits/{platform}/")
        return
    
    # Lista todos os arquivos JSON no diretório
    json_files = list(json_files_path.glob("*.json"))
    if not json_files:
        logging.error(f"Nenhum arquivo JSON encontrado em: {json_files_path}")
        return
    
    logging.info(f"Encontrados {len(json_files)} arquivo(s) JSON para processar")
    
    total_commits_found = 0
    valid_commits_processed = 0
    
    for json_file in json_files:
        logging.info(f"Processando arquivo: {json_file}")
        try:
            with open(json_file, "r", encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Erro de JSON ao ler {json_file}: {e}")
            continue
        except Exception as e:
            logging.error(f"Erro geral ao ler {json_file}: {e}")
            continue

        commits = data.get("commits", [])
        total_commits_found += len(commits)
        logging.info(f"Arquivo {json_file.name} contém {len(commits)} commits")
        
        for i, commit in enumerate(commits):
            message = commit.get("message")
            git_show = commit.get("git_show")
            commit_hash = commit.get("hash")

            # Debug detalhado para o primeiro commit problemático
            if i == 0:
                logging.info(f"Primeiro commit - message: {bool(message)}, git_show: {bool(git_show)}, hash: {bool(commit_hash)}")
                if message:
                    logging.info(f"Primeira linha da mensagem: {repr(message.split(chr(10))[0][:100])}")

            if not message:
                logging.debug(f"Commit sem mensagem ignorado: {commit_hash}")
                continue
                
            if not git_show:
                logging.debug(f"Commit sem git_show ignorado: {commit_hash}")
                continue
                
            # CORREÇÃO: Se não há hash no JSON, extrai do git_show
            if not commit_hash:
                commit_hash = extract_commit_hash_from_git_show(git_show)
                if not commit_hash or commit_hash == "unknown":
                    logging.debug(f"Commit sem hash válido ignorado")
                    continue

            cleaned_message = remove_prefix(message)
            cleaned_message = re.sub(r"(?i)breaking change:", "", cleaned_message).strip()
            
            # Usa a nova função de formatação
            try:
                formatted_git_show = format_git_show(git_show, cleaned_message)
            except Exception as e:
                logging.warning(f"Erro ao formatar git_show para commit {commit_hash}: {e}")
                continue
                
            label, is_breaking = classify_commit(message)
            
            # Usa deduplicação baseada em hash e input_text para evitar duplicatas
            commit_identifier = (commit_hash, formatted_git_show[:100])  # Primeiros 100 chars como identificador
            if commit_identifier not in seen_commits:
                seen_commits.add(commit_identifier)
                
                all_commits_data.append({
                    "input_text": formatted_git_show,
                    "label": label,
                    "message": message,
                    "breaking_change": is_breaking,
                    "hash": commit_hash,
                })
                valid_commits_processed += 1
    
    logging.info(f"Total de commits encontrados: {total_commits_found}")
    logging.info(f"Total de commits válidos processados: {valid_commits_processed}")
    
    if not all_commits_data:
        logging.warning(f"Nenhum commit válido encontrado para a plataforma '{platform}'.")
        logging.info("Verifique se os arquivos JSON contêm as chaves 'message', 'git_show' e 'hash'")
        return

    df = pd.DataFrame(all_commits_data)
    
    # Remove duplicatas baseado em hash (já foi feito durante o processamento, mas garantindo)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['hash']).reset_index(drop=True)
    final_count = len(df)
    
    if initial_count != final_count:
        logging.info(f"Removidas {initial_count - final_count} duplicatas adicionais")
    
    logging.info(f"Total de commits únicos para '{platform}': {final_count}")

    # --- Amostragem Estratificada Robusta ---
    population_size = len(df)
    desired_sample_size = calculate_sample_size(population_size, margin_error, confidence_level)
    
    logging.info(f"Tamanho da população: {population_size}")
    logging.info(f"Tamanho da amostra desejado: {desired_sample_size}")
    
    if desired_sample_size >= population_size:
        sampled_df = df
        logging.info("Usando toda a população (amostra >= população)")
    else:
        try:
            # Verifica se a estratificação é possível
            label_counts = df['label'].value_counts()
            min_count = label_counts.min()
            
            if min_count < 2:
                logging.warning("Alguns labels têm menos de 2 amostras. Realizando amostragem simples.")
                sampled_df = df.sample(n=desired_sample_size, random_state=42)
            else:
                # Usa train_test_split para amostragem estratificada
                _, sampled_df = train_test_split(
                    df,
                    test_size=desired_sample_size,
                    stratify=df['label'],
                    random_state=42
                )
                logging.info("Amostragem estratificada realizada com sucesso")
        except ValueError as e:
            logging.warning(f"Erro na amostragem estratificada: {e}. Realizando amostragem simples.")
            sampled_df = df.sample(n=min(desired_sample_size, len(df)), random_state=42)

    # Salva o dataset final
    output_file = DATA_PATH / "dataset" / f"E2_sample_{platform}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        sampled_df.to_json(output_file, orient="records", indent=4, force_ascii=False)
        logging.info(f"✅ Dataset criado com {len(sampled_df)} amostras para '{platform}' e salvo em '{output_file}'.")
    except Exception as e:
        logging.error(f"Erro ao salvar o dataset: {e}")
        return
    
    # Mostra distribuição dos labels
    label_distribution = sampled_df['label'].value_counts()
    logging.info(f"\n📊 Distribuição dos labels na amostra:")
    for label, count in label_distribution.items():
        logging.info(f"  {label}: {count}")


if __name__ == "__main__":
    platforms_to_process = ["npm"]
    MARGIN_ERROR = 0.05
    CONFIDENCE_LEVEL = 0.95
    
    for platform in platforms_to_process:
        process_platform(platform, MARGIN_ERROR, CONFIDENCE_LEVEL)