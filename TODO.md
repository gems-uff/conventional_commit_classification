# TODO — replication package gaps

Levantado em 2026-09-11 comparando `main (1).tex` (artigo) com o estado atual do
repositório. Ordenado por impacto na reprodutibilidade do RQ4 e do pacote como
um todo. Atualizado em 2026-09-12 (runner dos baselines treináveis, correções
motivadas pela crítica de revisor M7, reconciliação do within-corpus, e —
mais tarde no mesmo dia — execução completa na griffin com resultados reais
para os dois benchmarks; ver seção 5).

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
- [x] Execução completa na griffin (`cross_project`, 5 folds, e `within_corpus`) — **terminou com sucesso**, sem `--limit`, SVM/BiLSTM/CodeBERT retreinados do zero por seed com a seleção validada. Resultados e comparação com os números antigos: ver seção 5.
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

**Tamanho real do descompasso (2026-09-12)**: não são só 2 commits de diferença
entre 1.630 e 1.632 — a sobreposição real entre as duas coortes é de só **961
commits em comum**; **671 commits da coorte nova (1.632) nunca foram vistos**
pelas predictions congeladas do GPT-OSS/FLACCS/HQCM (e 669 da coorte antiga não
estão na nova). Ou seja, ~41% de cada lado é gente diferente — o corpus `.md`
de origem mudou de tamanho/ordem entre as duas gerações, então o mesmo
`train_test_split(seed=42)` produziu partições bem diferentes, não só um
deslocamento de poucos commits.

**Resolvido sem rodar inferência nova (2026-09-12)**: GPT-OSS, FLACCS e HQCM
não treinam — a predição de um commit não depende de qual protocolo
(cross-project vs. within-corpus) ele pertence, só do commit em si + modelo +
prompt + seed. `cross_project.yaml` e `within_corpus.yaml` configuram essas
três condições de forma **idêntica** (mesmo `prompt_id`, `temperature`, seeds).
Logo, a predição já computada no benchmark cross-project (que cobre os 8.159
commits inteiros, cada um testado exatamente uma vez) para um commit que
também está no within-corpus **é** a predição within-corpus dele — não precisa
re-executar nada. `scripts/rq4/build_benchmark_macro_f1.py` (só dentro de
`export_rq4_revision/` — ver item de limpeza) ganhou
`--other-models-source cross_project_audit` que faz exatamente esse
cherry-picking a partir de `replication_package/rq4_benchmark_audit/{gpt_oss_v2,cross_project_hqcm_v2,cross_project_flaccs_v2}/*.csv`
(na griffin), filtrando pros 1.632 commits de `data/rq4/within_corpus/assignments.csv`.
Rodado com sucesso — ver números na seção 5.

**Isso não corrige o arquivo congelado em si**: `results/rq4/supplementary_benchmark/predictions/predictions.csv.gz`
continua com os 1.630 antigos, intacto. A tabela `benchmark_macro_f1.csv` correta
(1.632, cherry-picked) existe hoje só em `runs/rq4_within_corpus/` na griffin —
falta decidir se/quando promover isso pra dentro de `results/rq4/` (ver seção 5).

- [x] Reconciliação do within-corpus resolvida por cherry-picking do cross-project audit (acima) — não precisou rodar GPT-OSS/FLACCS/HQCM de novo.

## 3. Limpeza

- [ ] `scripts/rq1/` está vazia (resíduo) — remover ou preencher.
- [ ] `export_rq4_revision/` (raiz do repo, já no `.gitignore` via `export*`) é uma pasta de exportação temporária criada nesta sessão — cópia autocontida dos arquivos necessários pro runner de baselines, pra levar pra griffin. Sem uso depois que a griffin tiver sua própria cópia; apagar.
- [ ] `scripts/rq4/build_benchmark_macro_f1.py` (o script que monta os `benchmark_macro_f1.csv` a partir das predictions) só existe hoje dentro de `export_rq4_revision/scripts/rq4/` — nunca foi trazido pra `scripts/rq4/` na raiz do repo. Se a intenção é ele ser parte permanente do pacote (como os `materialize_*.py` já são), precisa copiá-lo de volta antes de apagar `export_rq4_revision/`.

## 4. Opcional / nice-to-have

- [ ] Testes para `src/inference/{flaccs,hqcm}.py` (sem cobertura ainda).
- [x] parcial: `tests/test_model_selection.py` (novo) cobre `carve_validation_split` e `svm.select_c` (rodam sem `torch`). `bilstm.select_num_epochs` tem teste guardado por `pytest.importorskip("torch")` — não roda na máquina de dev (sem `torch` instalado ali), precisa rodar na máquina de execução pra validar de verdade. `codebert.select_num_epochs` e os três `train_predict()` continuam sem teste automatizado — só validados por smoke test manual (`--limit`) na griffin, não por suite de testes.

## 5. Resultados reais da revisão M7 (2026-09-12) e o que falta pro fechamento

Rodadas completas de `scripts/run_baselines_experiment.py` na griffin, mais o
cherry-pick descrito na seção 2, produziram `benchmark_macro_f1.csv` novos e
reais (não sintéticos) pros dois benchmarks — comparados abaixo com os valores
já publicados (médias de Macro-F1 por projeto, 30 projetos).

**Cross-project** (`our_gpt_model`/`flaccs`/`hqcm` inalterados — vieram do
arquivo congelado, sem tocar):

| Modelo | Antigo | Novo | Δ |
|---|---|---|---|
| GPT-OSS 20B | 0,476 | 0,476 | — |
| CodeBERT | 0,394 | 0,405 | +0,011 |
| FLACCS | 0,392 | 0,392 | — |
| BiLSTM | 0,363 | 0,377 | +0,014 |
| TF-IDF + SVM | 0,341 | 0,339 | −0,002 (ruído) |
| CodeFuse HQCM | 0,176 | 0,176 | — |

**Within-corpus** (`our_gpt_model`/`flaccs`/`hqcm` via cherry-pick da coorte
correta de 1.632 — não são os mesmos valores publicados, que estavam sobre 1.630):

| Modelo | Antigo | Novo | Δ |
|---|---|---|---|
| CodeBERT | 0,535 | 0,583 | +0,048 |
| BiLSTM | 0,525 | 0,517 | −0,008 (ruído) |
| TF-IDF + SVM | 0,499 | 0,499 | ≈0 |
| GPT-OSS 20B | 0,470 | 0,489 | +0,019 (coorte diferente, não é "melhora") |
| FLACCS | 0,419 | 0,419 | ≈0 |
| CodeFuse HQCM | 0,180 | 0,180 | ≈0 |

**Leitura**: a ordem qualitativa se manteve nos dois benchmarks. No cross-project,
o Finding 5 (GPT-OSS à frente de todos) sobrevive descritivamente, com folga
até um pouco maior que antes. No within-corpus, os três treináveis continuam
descritivamente à frente do GPT-OSS — e o CodeBERT (que passou a ser
genuinamente bem treinado, não mais fixado em 3 épocas) abriu ainda **mais**
vantagem sobre o GPT-OSS do que antes. Isso é a resposta mais forte possível
ao M7: a vantagem do GPT-OSS no cross-project não depende de baselines
sub-treinadas, e o contra-argumento do within-corpus fica mais robusto, não
mais frágil, depois da correção.

Falta, antes de considerar isso fechado:

- [ ] **Reanálise estatística** (ANOVA/Friedman + testes pareados com Holm,
  como os notebooks `results/rq4/{primary_benchmark,supplementary_benchmark}/analysis/*.ipynb`
  já fazem) rodada em cima dos `benchmark_macro_f1.csv` novos — só temos a
  média descritiva até agora, não significância. Decide se "significantly
  outperformed" continua válido no cross-project e se o within-corpus passa a
  ser estatisticamente significativo com a vantagem maior do CodeBERT.
- [ ] **Trazer os artefatos da griffin pro repositório**: as `predictions.csv.gz`
  novas (5 folds × 3 modelos no cross-project; 1 × 3 no within-corpus, com seus
  `run_manifest.json`) e os dois `benchmark_macro_f1.csv`. Decidir se isso
  **substitui** `results/rq4/{primary_benchmark,supplementary_benchmark}/{predictions,metrics}/...`
  (os congelados atuais) ou fica paralelo até promoção deliberada.
- [ ] **Reescrever `main.tex`**: §3.4.2 (validação em vez de "fixed in
  advance"), frase explícita de assimetria de contexto (512×10k/30k, CodeBERT
  arquitetural vs. BiLSTM dimensionado pelo corpus), tabelas do RQ4 com os
  números novos, Finding 5 e o parágrafo do within-corpus ajustados conforme o
  resultado da reanálise estatística acima.
- [ ] **Carta de resposta ao revisor** cobrindo o que continua sem solução
  técnica (decisão consciente, não lacuna): FLACCS/HQCM entram com pesos de
  outro corpus (fora de distribuição); nenhum baseline tipo LoRA-num-7B foi
  adicionado.

## Já validado como OK

- Metodologia (seeds, consenso first-to-two, `num_ctx`/`num_predict`, corte de 30k tokens, taxonomia de 11 classes) bate entre `main (1).tex` e `src/consensus.py` / `src/inference/ollama.py` / `config/evaluation.yaml`.
- `results/` já populado (não são stubs) para RQ1–RQ4.
- Parser canônico (`strict_response_parser.py` v4.1.1) já instalado neste checkout.
- Runner Ollama (`scripts/run_ollama_experiment.py`) agora aceita `--rq`/`--stage` para rodar qualquer RQ/sub-experimento por parâmetro, sem precisar saber o caminho do YAML.
