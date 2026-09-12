# TODO — replication package gaps

Levantado em 2026-09-11 comparando `main (1).tex` (artigo) com o estado atual do
repositório. Ordenado por impacto na reprodutibilidade do RQ4 e do pacote como
um todo. Atualizado em 2026-09-12 (runner dos baselines treináveis, correções
motivadas pela crítica de revisor M7, e reconciliação do within-corpus).

## 1. Faltam runners para os backends não-Ollama — baselines treináveis resolvidos; FLACCS/HQCM ainda faltam

`scripts/run_ollama_experiment.py` só executa GPT-OSS e as demais condições
generativas via Ollama. FLACCS e HQCM ainda não têm runner (módulos prontos em
`src/inference/{flaccs,hqcm}.py`, nenhum script os orquestra a partir de
`config/experiments/rq4/*.yaml`). Os baselines treináveis (TF-IDF+SVM, BiLSTM,
CodeBERT) **agora têm runner**, criado nesta sessão.

- [x] **`scripts/run_baselines_experiment.py`** (2026-09-12) — runner único para os três treináveis, cobrindo `cross_project.yaml` (5 folds) e `within_corpus.yaml` (holdout único). Decisão tomada sobre a pergunta em aberto abaixo: **um script só para os três treináveis**, separado dos runners de FLACCS/HQCM/Ollama (não um dispatcher único pra todos os backends, nem um script por modelo).
  - Reusa `src/consensus.py::resolve_first_to_two`/`AdaptiveSeedPolicy`, `src/data/loader.py`, `src/representations.py`; manifest/journal/`--resume` no mesmo padrão do `run_ollama_experiment.py`, resumível na granularidade "seed já completa para este fold" (uma seed = um retreino do zero + predição em lote, não por commit individual).
  - Hiperparâmetros originais confirmados corretos contra um pacote legado paralelo (`rq4_non_cross_v2_package`, fora deste repo: `run_svm.py`/`run_bilstm.py`/`run_codebert.py` + `train_*_multiseed.py`) que reimplementa o mesmo treino via scripts standalone.
  - Política de seed adaptativa (do `METHODOLOGY.md` desse pacote legado): quando um commit não fecha consenso após 10/20/30, treina-se um modelo **novo do zero** na base de treino **completa**; só prediz os commits ainda não resolvidos. Sem teto de seed (consistente com `AdaptiveSeedPolicy`, o `README.md` e o texto ativo do artigo). Validado empiricamente na execução real (griffin): seeds extras (40+) confirmadas genuinamente necessárias — nenhuma tinha 2-de-3 já batendo em seed 10/20/30, e `<invalid_output>` nunca aparece (esperado, são classificadores de conjunto fechado, sem geração de texto).
  - `consensus_manager.py`/`core/protocol.py` do pacote legado (reimplementação própria de first-to-two) não foram portados — o runner novo reusa só `src/consensus.py`.
  - Sem persistência de checkpoint de modelo treinado (decisão): só predictions + manifest sobrevivem a um run.
- [x] **Correção adicional motivada pela crítica de revisor M7** (hiperparâmetro/época fixados sem validação; assimetria de contexto 512×30k) — feita junto com o runner:
  - `src/training/model_selection.py` (novo): `carve_validation_split()`, split interno treino/dev (agrupado por projeto no cross-project, aleatório estratificado no within-corpus), nunca toca o teste.
  - `select_c()` (SVM) / `select_num_epochs()` (BiLSTM, CodeBERT): escolhem o hiperparâmetro/época uma vez por fold via essa validação, fixo depois entre as seeds — substitui os `C=1.0`/`num_epochs=10`/`num_train_epochs=3` antes fixados sem busca. Validado ao vivo na griffin: CodeBERT chegou a escolher 13 épocas (Macro-F1 de validação subindo de 0,14 na época 3 pra 0,29 na época 13).
  - `config/training/{svm,bilstm,codebert}.yaml` (novos — antes referenciados pelos YAMLs de experimento mas inexistentes no repo). `BiLstmConfig.max_seq_len` foi de 512 pra 10.000 (p95 empírico medido nesta sessão: 512 tokens truncava 57,6% do corpus pre-cutoff). CodeBERT continua em 512 — teto arquitetural do checkpoint (`microsoft/codebert-base`, `max_position_embeddings=514`), não corrigível por config.
  - **Ainda falta, e isso é trabalho de texto, não de código**: `main.tex` §3.4.2 ainda descreve "training schedules were fixed in advance"; falta a frase explícita sobre a assimetria 512×30k (CodeBERT é limite arquitetural, BiLSTM agora é escolha dimensionada pelo corpus); e o Finding 5 ("GPT-OSS significantly outperformed all five...") só pode ser reavaliado depois que a execução completa (abaixo) terminar e as métricas forem recalculadas.
  - Ver `tests/test_model_selection.py` para cobertura (parcial — item 4).
- [ ] Execução completa disparada em 2026-09-12 na griffin (`cross_project` + `within_corpus`, sem `--limit`) — em andamento no fim desta sessão, resultado ainda não conferido.
- [ ] `scripts/run_flaccs_experiment.py` (ou equivalente) para as condições FLACCS de `rq4/artifact_flaccs.yaml`.
- [ ] `scripts/run_hqcm_experiment.py` para HQCM (`rq4/artifact_hqcm.yaml` + protocolo original de 518 commits).

## 2. RQ4 within-corpus (Ollama) com preflight quebrado — preflight corrigido; predictions do GPT-OSS/FLACCS/HQCM ainda pendentes

Já documentado no `README.md`: os assignments atuais selecionam 1.632 commits
enquanto as predictions congeladas têm 1.630, com diferenças de membership.
`--rq rq4 --stage within_corpus` vai falhar até isso ser reconciliado a partir
da evidência original — o runner propositalmente não regenera o split nem
ignora isso com `--limit`.

**Preflight corrigido (2026-09-12)**: `config/experiments/rq4/within_corpus.yaml`
(`shared.expected_rows`/`shared.expected_cohort_sha256`) apontava para a coorte
antiga (1.630). Atualizado para `1632`/
`cc25059fa0b2597e5f07bef599158e45ae627d49562597bdcca486c936bc8256` — o valor
confirmado abaixo. `--rq rq4 --stage within_corpus` (Ollama) e
`scripts/run_baselines_experiment.py --stage within_corpus` (baselines
treináveis) agora passam no preflight; verificado com `--dry-run` em ambos.
**Isso não regenera nenhuma predição** — só para de bloquear novas execuções
contra a coorte correta. As predictions congeladas do GPT-OSS/FLACCS/HQCM em
`results/rq4/supplementary_benchmark/` continuam intactas (1.630, agora
sabidamente desatualizadas) até a ação abaixo ser feita.

**Direção confirmada**: o artigo (`main (1).tex`, linha 663) diz explicitamente
"6,527 training commits and 1,632 test commits". O pacote legado
`rq4_non_cross_v2_package` também usa 1.632 e trata a coorte de 1.630 como
"legacy... must not be mixed with this package". Logo, `1.632` (o que já está
em `data/rq4/within_corpus/assignments.csv`) é o valor correto do artigo — é
`results/rq4/supplementary_benchmark/predictions/predictions.csv.gz` (1.630)
que está desatualizado.

**Confirmado por hash (2026-09-12)**: comparado membro a membro, não só por
contagem. O `cohort_sha256` (mesmo algoritmo de `src/data/loader.py`) do lado
`split=="test"` de `data/rq4/within_corpus/assignments.csv` bate exatamente
com o de `dataset/rq4_v2_from_md_11classes/within_corpus/test.csv` do
repositório legado (`~/nrn` de origem):
`cc25059fa0b2597e5f07bef599158e45ae627d49562597bdcca486c936bc8256`,
1.632 linhas dos dois lados. Isso prova que `assignments.csv` é exatamente
reproduzível a partir do `.md` de origem (via `prepare_rq4_v2_splits.py`,
`test_size=0.2`, `random_state=42`) e contém os commits corretos — o
problema é só a `predictions.csv.gz` congelada ter sido gerada contra um
snapshot mais antigo do corpus (66,9% truncamento do HQCM à parte).

- [ ] Regenerar/atualizar as predictions congeladas em `results/rq4/supplementary_benchmark/` a partir da coorte de 1.632 commits (não ajustar `assignments.csv` para 1.630). Isso exige rodar GPT-OSS/FLACCS/HQCM de novo sobre esse cohort (`run_ollama_experiment.py --rq rq4 --stage within_corpus` mais os runners de FLACCS/HQCM) — não é resolvido pelos runners de baseline treináveis criados em `scripts/run_baselines_experiment.py`.

## 3. Limpeza

- [ ] `scripts/rq1/` está vazia (resíduo) — remover ou preencher.
- [ ] `export_rq4_revision/` (raiz do repo) é uma pasta de exportação temporária criada nesta sessão — cópia autocontida dos arquivos necessários pro runner de baselines, pra levar pra griffin. Sem uso depois que a griffin tiver sua própria cópia; apagar.

## 4. Opcional / nice-to-have

- [ ] Testes para `src/inference/{flaccs,hqcm}.py` (sem cobertura ainda).
- [x] parcial: `tests/test_model_selection.py` (novo) cobre `carve_validation_split` e `svm.select_c` (rodam sem `torch`). `bilstm.select_num_epochs` tem teste guardado por `pytest.importorskip("torch")` — não roda na máquina de dev (sem `torch` instalado ali), precisa rodar na máquina de execução pra validar de verdade. `codebert.select_num_epochs` e os três `train_predict()` continuam sem teste automatizado — só validados por smoke test manual (`--limit`) na griffin, não por suite de testes.

## Já validado como OK

- Metodologia (seeds, consenso first-to-two, `num_ctx`/`num_predict`, corte de 30k tokens, taxonomia de 11 classes) bate entre `main (1).tex` e `src/consensus.py` / `src/inference/ollama.py` / `config/evaluation.yaml`.
- `results/` já populado (não são stubs) para RQ1–RQ4.
- Parser canônico (`strict_response_parser.py` v4.1.1) já instalado neste checkout.
- Runner Ollama (`scripts/run_ollama_experiment.py`) agora aceita `--rq`/`--stage` para rodar qualquer RQ/sub-experimento por parâmetro, sem precisar saber o caminho do YAML.
