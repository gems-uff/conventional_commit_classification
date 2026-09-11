# TODO — replication package gaps

Levantado em 2026-09-11 comparando `main (1).tex` (artigo) com o estado atual do
repositório. Ordenado por impacto na reprodutibilidade do RQ4 e do pacote como
um todo.

## 1. Faltam runners para os backends não-Ollama (bloqueia reprodução completa do RQ4)

`scripts/run_ollama_experiment.py` só executa GPT-OSS e as demais condições
generativas via Ollama. FLACCS, HQCM, e os baselines treináveis (TF-IDF+SVM,
BiLSTM, CodeBERT) têm módulos prontos em `src/inference/{flaccs,hqcm}.py` e
`src/training/{svm,bilstm,codebert}.py`, mas nenhum script de linha de comando
os orquestra a partir de `config/experiments/rq4/*.yaml`. Só existem os
scripts de preparação de dados (`scripts/rq4/materialize_*.py`), não de
execução/treino.

- [ ] `scripts/run_flaccs_experiment.py` (ou equivalente) para as condições FLACCS de `rq4/artifact_flaccs.yaml`.
- [ ] `scripts/run_hqcm_experiment.py` para HQCM (`rq4/artifact_hqcm.yaml` + protocolo original de 518 commits).
- [ ] Runner para os baselines treináveis (SVM/BiLSTM/CodeBERT) cobrindo `cross_project.yaml` e `within_corpus.yaml`.
  - Hiperparâmetros já confirmados corretos: `src/training/{svm,bilstm,codebert}.py::train_predict()` bate 1:1 com um pacote legado paralelo (`rq4_non_cross_v2_package`, fora deste repo) que reimplementa o mesmo treino via scripts standalone (`run_svm.py`/`run_bilstm.py`/`run_codebert.py`).
  - Política de seed adaptativo (do `METHODOLOGY.md` desse pacote legado, adotada aqui): quando um commit não fecha consenso após 10/20/30, treina-se um modelo **novo do zero** (nova inicialização, mesma seed) na base de treino **completa**; ele só prediz os commits ainda não resolvidos. Não é treino incremental.
  - **Não** importar o teto de seed 80 desse pacote legado (lá, esgotar a seed 80 sem consenso virava `<invalid_output>`). Decisão: manter o runner atual sem teto, consistente com `src/consensus.py` (`AdaptiveSeedPolicy` sem máximo), o `README.md` ("No finite seed list determines exhaustion") e o texto ativo do artigo (Seção 3.3: seeds incrementados de 10 em 10 "until a terminal ... outcome is reached", sem menção a teto — o trecho com "40,50,60,70,80" no `.tex` está comentado/não faz parte do texto final).
  - O `consensus_manager.py`/`core/protocol.py` desse pacote legado reimplementam first-to-two e o parser por fora — não portar essa duplicação; o runner novo deve reusar `src/consensus.py::resolve_first_to_two`, `src/parsing/strict.py` e `src/data/loader.py` como já faz `run_ollama_experiment.py`.
- [ ] Decidir se isso vira um `scripts/run_experiment.py` único que despacha por backend (conforme sugerido na seção "Integration status" do `README.md`) ou scripts separados como o do Ollama.

## 2. RQ4 within-corpus (Ollama) com preflight quebrado

Já documentado no `README.md`: os assignments atuais selecionam 1.632 commits
enquanto as predictions congeladas têm 1.630, com diferenças de membership.
`--rq rq4 --stage within_corpus` vai falhar até isso ser reconciliado a partir
da evidência original — o runner propositalmente não regenera o split nem
ignora isso com `--limit`.

**Direção confirmada**: o artigo (`main (1).tex`, linha 663) diz explicitamente
"6,527 training commits and 1,632 test commits". O pacote legado
`rq4_non_cross_v2_package` também usa 1.632 e trata a coorte de 1.630 como
"legacy... must not be mixed with this package". Logo, `1.632` (o que já está
em `data/rq4/within_corpus/assignments.csv`) é o valor correto do artigo — é
`results/rq4/supplementary_benchmark/predictions/predictions.csv.gz` (1.630)
que está desatualizado.

- [ ] Regenerar/atualizar as predictions congeladas em `results/rq4/supplementary_benchmark/` a partir da coorte de 1.632 commits (não ajustar `assignments.csv` para 1.630).

## 3. Limpeza

- [ ] `scripts/rq1/` está vazia (resíduo) — remover ou preencher.

## 4. Opcional / nice-to-have

- [ ] Testes para `src/inference/{flaccs,hqcm}.py` e `src/training/*` (hoje só o pipeline Ollama tem cobertura em `tests/`).

## Já validado como OK

- Metodologia (seeds, consenso first-to-two, `num_ctx`/`num_predict`, corte de 30k tokens, taxonomia de 11 classes) bate entre `main (1).tex` e `src/consensus.py` / `src/inference/ollama.py` / `config/evaluation.yaml`.
- `results/` já populado (não são stubs) para RQ1–RQ4.
- Parser canônico (`strict_response_parser.py` v4.1.1) já instalado neste checkout.
- Runner Ollama (`scripts/run_ollama_experiment.py`) agora aceita `--rq`/`--stage` para rodar qualquer RQ/sub-experimento por parâmetro, sem precisar saber o caminho do YAML.
