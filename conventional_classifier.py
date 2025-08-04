import asyncio
import json
import logging
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from ollama import chat
import re
import aiofiles
from contextlib import asynccontextmanager
import signal
import sys
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import hashlib
from collections import deque
import math
import tiktoken
from concurrent.futures import ProcessPoolExecutor
import ast
import json, pathlib, pprint
import openai
from openai import OpenAI
import os

LOG_FILE = "classification.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),  
        logging.StreamHandler()  
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()



# Configurações
BATCH_SIZE = 50  
MAX_RETRIES = 3
RETRY_DELAY = 5
PLATFORM = "npm"
MODEL = "qwen2.5-coder:32b"
EXTENSION = "v7"
STRATEGY = "E2"
TEMPERATURE = 0.1
MAX_CONCURRENT_REQUESTS = 3
REQUEST_TIMEOUT = 60
CHECKPOINT_INTERVAL = 3  
MAX_CONTENT_LENGTH = 750  
MAX_FAILED_ENTRIES = 5_000
enc = tiktoken.get_encoding("cl100k_base")  
process_pool = ProcessPoolExecutor(max_workers=2)
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
VALID_TYPES = {"fix", "feat", "build", "chore", "ci", "docs", "style", "refactor", "perf", "test", "others"}
CONVENTIONAL_COMMITS_PATTERN = re.compile(r"^([a-z]+)(\([^)]+\))?:\s*", re.IGNORECASE)
TYPE_RE = re.compile(r"type\s*[:\-]\s*([a-zA-Z_]+)", re.IGNORECASE)
REASON_RE = re.compile(r"reason[:\s\-]+(.+)", re.IGNORECASE)
BREAKING_RE = re.compile(r"breaking[_ ]change[:\s\-]+(true|false)", re.IGNORECASE)

# Arquivos
input_file = Path(f"data/dataset/{STRATEGY}_sample_{PLATFORM}.json")
output_file = Path(f"data/output/{MODEL}/{PLATFORM}/{STRATEGY}_{EXTENSION}.json")
checkpoint_file = Path(f"checkpoints/checkpoint_{PLATFORM}_{MODEL}_{EXTENSION}.json")
failed_commits_file = Path(f"data/failed/{MODEL}/{PLATFORM}/{EXTENSION}.json")
PROMPT_DIR = Path("prompts")

logger.info(f"🧪 Modelo selecionado: {MODEL}")


# Criar diretórios
for path in [output_file, checkpoint_file, failed_commits_file]:
    path.parent.mkdir(parents=True, exist_ok=True)

def load_prompt(extension: str, commit_input: str):
    """Carrega prompt template com verificação de erros"""
    prompt_path = PROMPT_DIR / f"{extension}.json"
    
    if not prompt_path.exists():
        logger.error(f"❌ Arquivo de prompt não encontrado: {prompt_path}")
        # Fallback para prompt básico
        return [
            {"role": "system", "content": "You are a commit classifier. Classify the commit type and provide a reason."},
            {"role": "user", "content": f"Classify this commit: {commit_input}"}
        ], []
    
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"❌ Erro ao decodificar JSON do prompt: {e}")
        return [
            {"role": "system", "content": "You are a commit classifier. Classify the commit type and provide a reason."},
            {"role": "user", "content": f"Classify this commit: {commit_input}"}
        ], []
    
    # Substitui placeholder apenas na última mensagem do user
    if messages and len(messages) > 0:
        messages[-1]["content"] = messages[-1]["content"].replace("{{COMMIT_INPUT}}", commit_input)
    
    # prompt_template = cópia do messages sem a última mensagem (sem o commit)
    prompt_template = messages[:-1] if len(messages) > 1 else []
    
    return messages, prompt_template

class AdaptiveSemaphore:
    def __init__(self, min_limit=1, max_limit=10, horizon=30, target_ratio=0.25):
        self._sem = asyncio.Semaphore(min_limit)
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.horizon = horizon
        self.target_ratio = target_ratio
        self.latencies = deque(maxlen=horizon)

    async def acquire(self):
        await self._sem.acquire()

    def release(self):
        self._sem.release()

    def record_latency(self, seconds: float):
        self.latencies.append(seconds)
        if len(self.latencies) == self.horizon:
            self._maybe_adjust()

    def _maybe_adjust(self):
        avg = sum(self.latencies) / len(self.latencies)
        if avg < REQUEST_TIMEOUT * self.target_ratio and self._sem._value < self.max_limit:
            self._sem._value += 1
            logger.info(f"🚀 Latência {avg:.2f}s → aumentando concorrência para {self._sem._value}")
        elif avg > REQUEST_TIMEOUT * (self.target_ratio * 2) and self._sem._value > self.min_limit:
            self._sem._value -= 1
            logger.info(f"🐢 Latência {avg:.2f}s → reduzindo concorrência para {self._sem._value}")

semaphore = AdaptiveSemaphore(min_limit=1, max_limit=6)

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

@dataclass
class CommitResult:
    input_text: str
    type: str
    reason: str
    raw_response: str
    label: str
    correct: bool
    breaking_change: bool
    expected_breaking_change: bool
    breaking_correct: bool
    processing_time: float
    commit_hash: str
    timestamp: str

@dataclass
class ProcessingStats:
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    start_time: datetime = None
    
    def to_dict(self) -> Dict:
        return {
            "total_processed": self.total_processed,
            "successful": self.successful,
            "failed": self.failed,
            "start_time": self.start_time.isoformat() if self.start_time else None
        }

def _md5_first_8(text):
    return hashlib.md5(text.encode()).hexdigest()[:8]

def is_openai_model(model_name: str) -> bool:
    return model_name.startswith("gpt")

class RateLimiter:
    def __init__(self, requests_per_second: float = 2.0):
        self.rate = requests_per_second
        self.last_request = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self.last_request
            wait_time = (1.0 / self.rate) - time_since_last
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            self.last_request = asyncio.get_event_loop().time()

rate_limiter = RateLimiter(1.0) 

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):  
            return obj.isoformat()     
        return super().default(obj)

def prepare_commit_content(input_text: str, max_tokens: int = MAX_CONTENT_LENGTH) -> str:
    if count_tokens(input_text) <= max_tokens:
        return input_text

    lines = input_text.splitlines()
    selected = []
    current_tokens = 0

    for line in lines:
        tok = count_tokens(line + "\n")
        if current_tokens + tok > max_tokens:
            break
        selected.append(line)
        current_tokens += tok

    selected.append(f"\n[TRUNCATED to {max_tokens} tokens]")
    return "\n".join(selected)

async def test_ollama_connection():
    """Testa se o Ollama está funcionando"""
    try:
        logger.info("🔍 Testando conexão com Ollama...")
        
        # Teste simples
        test_response = await asyncio.to_thread(
            chat,
            model=MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            options={"temperature": 0.1}
        )
        
        logger.info(f"✅ Ollama respondeu: {test_response['message']['content'][:50]}...")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao conectar com Ollama: {e}")
        logger.error(f"❌ Verifique se o Ollama está rodando e se o modelo '{MODEL}' está disponível")
        return False

async def classify_with_openai(messages, model="gpt-4", temperature=0.1, timeout=60):
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"❌ Erro na chamada da OpenAI: {e}")
        raise


async def classify_commit_with_retry(
        input_text: str,
        idx: int = 0,
        batch_index: int = 0
) -> Optional[Tuple[str, str, bool, str, float]]:
    """
    Classifica um commit usando o modelo do Ollama, com:
      • controle de concorrência (AdaptiveSemaphore)
      • rate‑limiting
      • até MAX_RETRIES tentativas com back‑off exponencial
      • logs detalhados
    Retorna (commit_type, reason, breaking_change, raw_response, processing_time)
    ou None se falhar em todas as tentativas.
    """
    start_time = asyncio.get_event_loop().time()

    # ----- controle de concorrência -----
    await semaphore.acquire()
    try:
        # ----- rate‑limit global -----
        await rate_limiter.acquire()

        prepared_input = prepare_commit_content(input_text)
        messages, _ = load_prompt(EXTENSION, prepared_input)

        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"🔄 Tentativa {attempt + 1}/{MAX_RETRIES} – commit {batch_index + idx}")

                # chamada síncrona ao Ollama em thread separada + timeout
                if "gpt" in MODEL.lower():
                    raw_response = await asyncio.wait_for(
                        classify_with_openai(messages, temperature=TEMPERATURE, timeout=REQUEST_TIMEOUT),
                        timeout=REQUEST_TIMEOUT
                    )
                else:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            chat,
                            model=MODEL,
                            messages=messages,
                            options={"temperature": TEMPERATURE}
                        ),
                        timeout=REQUEST_TIMEOUT
                    )
                    raw_response = response["message"]["content"]

                logger.debug(f"✅ Resposta do modelo (commit {batch_index + idx}): "
                             f"{raw_response[:120].replace(chr(10), ' ')}…")

                commit_type, reason, breaking_change = parse_classification_response(raw_response)
                processing_time = asyncio.get_event_loop().time() - start_time

                # registra latência para adaptação do semáforo
                semaphore.record_latency(processing_time)

                return commit_type, reason, breaking_change, raw_response, processing_time

            except asyncio.TimeoutError:
                logger.warning(f"⏰ Timeout no commit {batch_index + idx} "
                               f"(tentativa {attempt + 1}/{MAX_RETRIES})")
            except Exception:
                logger.exception(f"❌ Erro processando commit {batch_index + idx} "
                                 f"(tentativa {attempt + 1}/{MAX_RETRIES})")

            # back‑off exponencial antes da próxima tentativa (se houver)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))

        logger.error(f"💥 Falha definitiva no commit {batch_index + idx}")
        return None

    finally:
        # libera o slot no semáforo mesmo em caso de erro
        semaphore.release()


def parse_classification_response(text: str) -> Tuple[str, str, bool]:
    """Versão robusta com logs e fallback seguro."""
    commit_type = "others"
    reason = "No reason provided"
    breaking_change = False

    try:
        type_match = TYPE_RE.search(text)
        reason_match = REASON_RE.search(text)
        breaking_match = BREAKING_RE.search(text)

        if type_match:
            raw_type = type_match.group(1).lower().strip()
            if raw_type in VALID_TYPES:
                commit_type = raw_type
            else:
                logger.warning(f"⚠️ Tipo inválido identificado: '{raw_type}'. Substituído por 'others'.")
        else:
            logger.warning("⚠️ Campo 'type' não encontrado na resposta.")

        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            logger.warning("⚠️ Campo 'reason' não encontrado na resposta.")

        if breaking_match:
            breaking_change = breaking_match.group(1).strip().lower() == "true"
        else:
            logger.warning("⚠️ Campo 'breaking_change' não encontrado na resposta.")

        # Tenta extrair JSON se presente
        try:
            json_blocks = re.findall(r"```json\s*({.*?})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
            if json_blocks:
                parsed_json = json.loads(json_blocks[0])
                commit_type = parsed_json.get("type", parsed_json.get("Type", commit_type)).lower()
                reason = parsed_json.get("reason", parsed_json.get("Reason", reason))
                breaking_change = parsed_json.get("breaking_change", parsed_json.get("Breaking_Change", breaking_change))

                if commit_type not in VALID_TYPES:
                    logger.warning(f"⚠️ Tipo inválido no JSON: '{commit_type}'")
                    commit_type = "others"

        except Exception as e:
            logger.debug(f"Erro ao tentar extrair JSON formatado: {e}")

    except Exception as e:
        logger.error(f"❌ Erro ao fazer parse da resposta LLM: {e}")
        logger.debug(f"Resposta problemática:\n{text}")

    return commit_type, reason, breaking_change

async def save_checkpoint_async(stats: ProcessingStats, current_index: int):
    """Salva checkpoint de forma assíncrona"""
    checkpoint_data = {
        "current_index": current_index,
        "stats": {
            "total_processed": stats.total_processed,
            "successful": stats.successful,
            "failed": stats.failed,
            "start_time": stats.start_time.isoformat() if stats.start_time else None
        },
        "timestamp": datetime.now().isoformat(),
        "version": EXTENSION
    }
    
    try:
        async with aiofiles.open(checkpoint_file, 'w') as f:
            await f.write(json.dumps(checkpoint_data, indent=2))
        logger.debug(f"💾 Checkpoint salvo: índice {current_index}")
    except Exception as e:
        logger.error(f"Erro salvando checkpoint: {e}")

async def load_existing_results() -> Tuple[List[Dict], List[str], List[str], List[bool], List[bool]]:
    """Carrega resultados existentes do arquivo de saída"""
    results = []
    y_true, y_pred = [], []
    expected_breaking_changes, predicted_breaking_changes = [], []
    
    if not output_file.exists():
        return results, y_true, y_pred, expected_breaking_changes, predicted_breaking_changes
    
    try:
        async with aiofiles.open(output_file, "r") as f:
            content = await f.read()
            existing = json.loads(content)
            existing_results = existing.get("results", [])
            
            for r in existing_results:
                if r.get("type") and r.get("label"):
                    results.append(r)
                    y_true.append(r["label"])
                    y_pred.append(r["type"])
                    expected_breaking_changes.append(r.get("expected_breaking_change", False))
                    predicted_breaking_changes.append(r.get("breaking_change", False))
        
        logger.info(f"📄 Carregados {len(results)} resultados existentes")
                    
    except Exception as e:
        logger.error(f"Erro carregando resultados existentes: {e}")
    
    return results, y_true, y_pred, expected_breaking_changes, predicted_breaking_changes

async def load_checkpoint_async() -> Tuple[int, ProcessingStats]:
    """Carrega checkpoint de forma assíncrona"""
    if not checkpoint_file.exists():
        return 0, ProcessingStats(start_time=datetime.now())
    
    try:
        async with aiofiles.open(checkpoint_file, 'r') as f:
            content = await f.read()
            data = json.loads(content)
            
        stats_data = data.get("stats", {})
        
        start_time_str = stats_data.get("start_time")
        if start_time_str and start_time_str != "null":
            try:
                stats_data["start_time"] = datetime.fromisoformat(start_time_str)
            except:
                stats_data["start_time"] = datetime.now()
        else:
            stats_data["start_time"] = datetime.now()

        stats = ProcessingStats(**stats_data)
        current_index = data.get("current_index", 0)
        
        logger.info(f"📂 Checkpoint carregado: retomando do índice {current_index}")
        logger.info(f"📊 Stats carregados: {stats.successful} sucessos, {stats.failed} falhas")

        return current_index, stats
        
    except Exception as e:
        logger.error(f"Erro carregando checkpoint: {e}")
        return 0, ProcessingStats(start_time=datetime.now())

async def save_failed_commit(commit_data: Dict, error_info: str):
    """Salva commits que falharam para análise posterior"""
    commit_hash = commit_data.get("hash", "unknown")

    failed_data = {
        "commit_hash": commit_hash,
        "input_text": commit_data.get("input_text", "")[:500],  
        "label": commit_data.get("label"),
        "error": error_info,
        "timestamp": datetime.now().isoformat()
    }
    
    failed_commits = []
    if failed_commits_file.exists():
        try:
            async with aiofiles.open(failed_commits_file, 'r') as f:
                content = await f.read()
                failed_commits = json.loads(content)
        except:
            failed_commits = []
    
    failed_commits.append(failed_data)

    if len(failed_commits) > MAX_FAILED_ENTRIES:
        failed_commits = failed_commits[-MAX_FAILED_ENTRIES:]
    
    async with aiofiles.open(failed_commits_file, 'w') as f:
        await f.write(json.dumps(failed_commits, indent=2))

# Variáveis globais para o signal handler
global_stats = ProcessingStats()
global_current_index = 0

def signal_handler(signum, frame):
    logger.info("🛑 Interrupção recebida. Salvando estado…")
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.run_until_complete(save_checkpoint_async(global_stats, global_current_index))
    except Exception as e:
        logger.warning(f"Falha ao salvar checkpoint no handler: {e}")

    logger.info("✅ Checkpoint salvo. Encerrando.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def calculate_metrics(y_true, y_pred):
    classes = ['feat', 'fix', 'docs', 'style', 'refactor', 'perf', 'test', 'build', 'ci', 'chore', 'others']

    valid_indices = [i for i, pred in enumerate(y_pred) if pred != "Unknown"]
    if not valid_indices:
        return {
            'overall_accuracy': 0.0,
            'class_metrics': {
                cls: dict(precision=0.0, recall=0.0, f1_score=0.0, support=0)
                for cls in classes
            }
        }

    y_true_filtered = [y_true[i].lower() for i in valid_indices]
    y_pred_filtered = [y_pred[i].lower() for i in valid_indices]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_filtered, y_pred_filtered, labels=classes, zero_division=0
    )
    accuracy = float(accuracy_score(y_true_filtered, y_pred_filtered))

    return {
        'overall_accuracy': accuracy,
        'class_metrics': {
            cls: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            } for i, cls in enumerate(classes)
        }
    }

def calculate_binary_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

async def process_batch_concurrent(batch: List[Dict], batch_index: int) -> List[CommitResult]:
    """Processa um batch de commits de forma concorrente"""
    results = []

    async def process_one(commit: Dict, idx: int):
        input_text = commit.get('input_text')
        if not input_text:
            logger.warning(f"⚠️ Commit {batch_index+idx} sem input_text")
            return None

        classification_result = await classify_commit_with_retry(input_text, idx, batch_index)
        if classification_result is None:
            await save_failed_commit(commit, "Classification failed after retries")
            return None

        commit_type, reason, breaking_change, raw_response, processing_time = classification_result
        label = commit.get('label', '').lower()
        commit_type = commit_type.lower()
        correct = commit_type == label

        expected_breaking = commit.get("breaking_change", False)
        breaking_correct = breaking_change == expected_breaking
        commit_hash = commit.get("hash", "unknown")

        return CommitResult(
            input_text=input_text,
            type=commit_type,
            reason=reason,
            raw_response=raw_response,
            label=label,
            correct=correct,
            breaking_change=breaking_change,
            expected_breaking_change=expected_breaking,
            breaking_correct=breaking_correct,
            processing_time=processing_time,
            commit_hash=commit_hash,
            timestamp=datetime.now().isoformat()
        )

    tasks = [process_one(commit, idx) for idx, commit in enumerate(batch)]
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(task_results):
        if isinstance(result, CommitResult):
            results.append(result)
        elif isinstance(result, Exception):
            logger.error(f"❌ Exceção ao processar commit {batch_index+i}: {result}")
            await save_failed_commit(batch[i], str(result))

    return results

async def main():
    """Função principal melhorada com teste de conectividade"""
    global global_stats, global_current_index
    
    logger.info("🚀 Iniciando classificação de commits...")
    
    # Testa conectividade do Ollama primeiro
    if not is_openai_model(MODEL):
        if not await test_ollama_connection():
            logger.error("❌ Não foi possível conectar ao Ollama. Verifique:")
            logger.error("   1. Se o serviço está rodando: ollama serve")
            logger.error(f"   2. Se o modelo está disponível: ollama list | grep {MODEL}")
            logger.error(f"   3. Se necessário, baixe o modelo: ollama pull {MODEL}")
            return
    
    # Carrega dados
    try:
        async with aiofiles.open(input_file, 'r') as f:
            content = await f.read()
            data = json.loads(content)
    except FileNotFoundError:
        logger.error(f"❌ Arquivo de entrada não encontrado: {input_file}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"❌ Erro ao decodificar JSON: {e}")
        return
    
    # Carrega checkpoint
    start_index, stats = await load_checkpoint_async()
    global_stats = stats
    global_current_index = start_index
    
    # Carrega resultados existentes
    results, y_true, y_pred, expected_breaking_changes, predicted_breaking_changes = await load_existing_results()

    total_commits = len(data)
    logger.info(f"📊 Total: {total_commits} commits, retomando do índice: {start_index}")
    
    if start_index >= total_commits:
        logger.info("✅ Todos os commits já foram processados!")
        return

    # Processamento principal
    for i in range(start_index, total_commits, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_commits)
        batch = data[i:batch_end]
        
        logger.info(f"📦 Processando batch {i//BATCH_SIZE + 1}: commits {i}-{batch_end}")
        
        # Processa batch
        batch_results = await process_batch_concurrent(batch, i)
        
        # Atualiza stats
        global_stats.total_processed += len(batch)
        global_stats.successful += len(batch_results)
        global_stats.failed += len(batch) - len(batch_results)
        global_current_index = batch_end
        
        # Atualiza resultados
        for result in batch_results:
            results.append(asdict(result))
            y_true.append(result.label)
            y_pred.append(result.type)
            expected_breaking_changes.append(result.expected_breaking_change)
            predicted_breaking_changes.append(result.breaking_change)

        # Salva resultados
        if len(batch_results) > 0:
            metrics = {
                "label_metrics": calculate_metrics(y_true, y_pred),
                "breaking_change_metrics": calculate_binary_metrics(expected_breaking_changes, predicted_breaking_changes),
                "processing_stats": global_stats.to_dict()
            }
            
            _, prompt_template = load_prompt(EXTENSION, "placeholder")
            output_data = {
                "results": results,
                "metrics": metrics,
                "metadata": {
                    "model": MODEL,
                    "platform": PLATFORM,
                    "version": EXTENSION,
                    "last_updated": datetime.now().isoformat(),
                    "prompt_template": prompt_template
                }
            }
            
            async with aiofiles.open(output_file, 'w') as f:
                await f.write(json.dumps(output_data, indent=2, cls=NumpyEncoder))
        
        # Salva checkpoint
        await save_checkpoint_async(global_stats, batch_end)
        
        # Log de progresso
        accuracy = metrics["label_metrics"]["overall_accuracy"] if batch_results else 0
        success_rate = (global_stats.successful / global_stats.total_processed) * 100 if global_stats.total_processed > 0 else 0
        
        logger.info(f"✅ Batch concluído | Processados: {global_stats.total_processed} | "
                   f"Sucessos: {global_stats.successful} | Falhas: {global_stats.failed} | "
                   f"Taxa de sucesso: {success_rate:.1f}% | Accuracy: {accuracy:.3f}")

        await asyncio.sleep(0.1) 

    
    try:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("🗑️ Checkpoint removido - processamento completo")
    except Exception as e:
        logger.warning(f"Aviso: não foi possível remover checkpoint: {e}")

    
    total_time = (datetime.now() - stats.start_time).total_seconds()
    
    logger.info("🎯 Processamento concluído!")
    logger.info(f"⏱️  Tempo total: {total_time:.1f}s")
    logger.info(f"✅ Commits processados: {stats.total_processed}")
    logger.info(f"🎯 Commits classificados: {stats.successful}")
    logger.info(f"❌ Commits que falharam: {stats.failed}")
    logger.info(f"📊 Taxa de sucesso: {(stats.successful/stats.total_processed)*100:.1f}%")
    
    if y_true and y_pred:
        final_accuracy = calculate_metrics(y_true, y_pred)["overall_accuracy"]
        logger.info(f"🎯 Acurácia final: {final_accuracy:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
