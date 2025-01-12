import os
import datetime
from enum import Enum
from pathlib import Path

# Função para determinar o diretório do projeto
def project_root() -> Path:
    return Path(__file__).resolve().parent  

# Define os caminhos dos diretórios de interesse
DATA_PATH: Path = project_root() / 'data'
CORPUS_PATH: Path = project_root() / 'corpus'
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
REPO_PATH = os.path.expanduser("~/Documents/repos")
TMP_PATH: Path = project_root() / 'tmp'
CPU: int = int(os.getenv('CPU', 10))

RAPID_RELEASE_LIM = 7*6
TRAD_RELEASE_LIM = 7*12

class CycleType(Enum):
    FIRST = 'f'
    RAPID = 'r'
    TRAD = 't'
    OTHER = 'o'
    PROBLEM = 'p'

# Função para converter diferença de tempo em dias
def delta2days(td):
    return td.total_seconds() / datetime.timedelta(days=1).total_seconds()

# Função para obter o tempo de lançamento de uma versão
def get_release_time(srelease):
    if srelease.commits:
        return srelease.release.tag.commit.committer_time
    else:
        return srelease.time
    
# Função para obter o ciclo de desenvolvimento
def get_cycle(srelease):
    if srelease.prev_main_release:
        prev_release_time = get_release_time(srelease.prev_main_release)
        release_time = get_release_time(srelease)
        return release_time - prev_release_time
    else:
        return datetime.timedelta(0)

# Função para obter o início do desenvolvimento da versão
def get_start(srelease):
    if srelease.commits:
        return srelease.commits.first(lambda c: c.author_time).author_time
    else:
        return get_release_time(srelease)
        
# Função para obter a duração do desenvolvimento
def get_duration(srelease):
    return get_release_time(srelease) - get_start(srelease)
        
# Função para obter o atraso de início em relação à versão anterior
def get_start_delay(srelease):
    if srelease.prev_main_release:
        prev_release = srelease.prev_main_release
        return get_start(srelease) - get_release_time(prev_release)
    else:
        return datetime.timedelta(0)

# Função para obter os commits feitos antes da versão anterior
def get_commits_in_advance(srelease):
    if not srelease.commits or not srelease.prev_main_release:
        return 0
    
    prev_release = srelease.prev_main_release
    return [commit for commit in srelease.commits if commit.author_time < get_release_time(prev_release)]

# Funções para determinar o tipo de lançamento
def is_rapid_release(release):    
    if not release.commits:
        return False
    if not release.prev_main_release:
        return False
    days = delta2days(get_cycle(release))
    if days <= 0:
        return False
    if days > RAPID_RELEASE_LIM:
        return False
    return True

def is_trad_release(release):
    if not release.commits:
        return False
    if not release.prev_main_release:
        return False
    days = delta2days(get_cycle(release))
    if days <= 0:
        return False
    if days < TRAD_RELEASE_LIM:
        return False
    return True
