# TODO — replication package gaps

Levantado em 2026-09-11 comparando `main (1).tex` (artigo) com o estado atual do
repositório. Ordenado por impacto na reprodutibilidade do RQ4 e do pacote como
um todo. Atualizado em 2026-09-12 (runner dos baselines treináveis, correções
motivadas pela crítica de revisor M7, reconciliação do within-corpus, e —
mais tarde no mesmo dia — execução completa na griffin com resultados reais
para os dois benchmarks; ver seção 5). Revisado em 2026-09-24: confirmado por
inspeção direta dos arquivos que a promoção dos artefatos da griffin pra
`results/rq4/` e a reescrita de `main.tex` (seção 5) já tinham sido feitas,
mas continuavam marcadas como pendentes aqui — corrigido abaixo. Também nesta
revisão: `scripts/run_flaccs_experiment.py` (novo, item 1) fecha a lacuna do
runner FLACCS; HQCM foi decidido como limitação aceita, não lacuna a fechar
(ver item 1).

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
- [x] **`scripts/run_flaccs_experiment.py`** (2026-09-24) — runner dedicado à condição `flaccs` de `config/experiments/rq4/artifact_flaccs.yaml` (o artefato FLACCS liberado, `0x404/ccs-code-llama-7b`, no subconjunto de 2.573 commits compatíveis). GPT-OSS dessa mesma condição já rodava via `run_ollama_experiment.py --rq rq4 --stage artifact_flaccs`; a condição `flaccs` extended-11class de `cross_project.yaml`/`within_corpus.yaml` continua sem runner próprio porque já está congelada (item 2) — não é a mesma lacuna.
  - Mesmo padrão de manifest/journal/`--resume`/`--dry-run`/`--list-jobs` dos outros runners; reusa `src/consensus.py` (adaptive first-to-two) e `src/parsing/strict.py` (parser canônico v4.1.1) — não a extração por regex nem o voto por maioria simples dos scripts standalone legados (`flaccs_hf_len1024_multiseed_adaptive.py`, `flaccs_test_csv_classifier.py`) que este runner substitui.
  - O prompt (`p_flaccs_native_10class`) é resolvido via `prompts/manifest.yaml`, nunca hardcoded no script. `FlaccsBackend.build_prompt()` (novo, em `src/inference/flaccs.py`) faz o truncamento em nível de tokenizer (mensagem até `message_max_tokens=64`, diff absorvendo o restante de `context_window_tokens=1024`) a partir dos dois placeholders do template — difere um pouco do script legado `flaccs_hf_len1024_multiseed_adaptive.py`, que também cobrava alguns tokens do texto literal de wrapper ("- given commit message:") contra o teto da mensagem; a diferença é de poucos tokens em um orçamento de 1.024 e não muda quais commits precisam de truncamento (`data/rq4/artifact_compatible/flaccs/assignments.csv` já restringe a condição a commits cujo prompt original e não-truncado cabe no orçamento).
  - `do_sample: false` (greedy determinístico, sem temperature/top_p) vem do próprio `artifact_flaccs.yaml`, batendo com o que `FlaccsBackend` já implementava antes deste runner existir.
  - Validado nesta sessão: `--list-jobs`/`--dry-run` rodam limpos na máquina de desenvolvimento (sem `torch`/`transformers` instalados, como já documentado); `tests/test_flaccs_runner.py` (novo, 9 testes) cobre expansão de job, dry-run offline, run/resume básico, progressão de seed adaptativa, terminalidade por `<invalid_output>`, proteção de `results/`, e rejeição de identidade — tudo com um `FlaccsBackend` mockado (não precisa do modelo real). A suíte completa (`pytest`) continua 100% verde (88 passed, 1 skipped) depois da mudança.
  - **Ainda falta, e só pode ser feito na griffin**: a execução real (baixar `0x404/ccs-code-llama-7b` do Hugging Face, rodar sem `--limit`) e a decisão de promover o `predictions.csv.gz` resultante pra dentro de `results/rq4/related_work_artifacts/flaccs/` (hoje só tem a versão congelada de uma execução anterior).
- [x] **HQCM decidido como limitação aceita, não lacuna de código** (2026-09-24) — não haverá `scripts/run_hqcm_experiment.py`. Ao contrário do FLACCS (pesos prontos no Hugging Face, só inferência), reproduzir HQCM exigiria reconstruir a infraestrutura de fine-tuning do artefato original (`run_hqcm(1).py` legado precisa de `--hqcm-root` com um venv próprio e o adapter `chcl_llama2_7b_len1024` já treinado) — não é "rodar um modelo do Hub", é recriar o ambiente de treino. `results/rq4/related_work_artifacts/hqcm*/` e as colunas `hqcm`/`CodeFuse HQCM` de `cross_project`/`within_corpus` continuam com as predictions congeladas da execução original. Isso deve constar explicitamente na carta de resposta ao revisor (ver item já registrado abaixo sobre "FLACCS/HQCM entram com pesos de outro corpus").

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

**Reanálise estatística concluída (2026-09-14, rodada pelo Rodrigo fora desta
sessão, seguindo a mesma regra de decisão pré-especificada dos notebooks)**:

- *Cross-project (primary)*: Shapiro-Wilk rejeita normalidade só pro CodeBERT
  (W=.919, p=.026 — quase idêntico ao já publicado W=.921, p=.029) → rota
  Friedman, igual antes. χ²(5)=93,45, p<.001, Kendall's W=.623. Estrutura
  pareada **idêntica** à já publicada: GPT-OSS bate os cinco outros com
  significância; CodeBERT e FLACCS batem TF-IDF+SVM; CodeBERT vs FLACCS não
  significativo; BiLSTM não difere de TF-IDF/CodeBERT/FLACCS; HQCM perde pra
  todos. **O Finding 5 sobrevive intacto — nem uma comparação de significância
  mudou**, apesar do CodeBERT/BiLSTM terem melhorado descritivamente.
- *Within-corpus (supplementary)*: Shapiro-Wilk não rejeita normalidade em
  nenhum → rota paramétrica (ANOVA + Greenhouse-Geisser, Mauchly rejeitou
  esfericidade). F(3,52; 101,94)=71,78, p<.001, η²G=.544. **Achado novo**: o
  CodeBERT agora bate **todos os outros cinco modelos com significância**,
  incluindo o GPT-OSS (Hedges' g=0,64, médio, p_Holm=.013) — antes da correção,
  a vantagem descritiva do CodeBERT sobre o GPT-OSS não era significativa.
  BiLSTM e TF-IDF+SVM continuam à frente do GPT-OSS só descritivamente, sem
  significância (como antes).

Falta, antes de considerar isso fechado:

- [x] Reanálise estatística (ANOVA/Friedman + pareados com Holm) — feita pros
  dois benchmarks, achados acima.
- [x] **Trazer os artefatos da griffin pro repositório** (verificado em
  2026-09-24): `results/rq4/{primary_benchmark,supplementary_benchmark}/{predictions,metrics}/...`
  já contêm os valores novos (conferido célula a célula contra a tabela acima:
  CodeBERT 0,405/0,583, BiLSTM 0,377/0,517 etc.) — os congelados antigos foram
  substituídos, não ficaram paralelos.
- [x] **Reescrever `main.tex`** (verificado em 2026-09-24 contra `main(1).tex`,
  2.558 linhas): §3.4.2 já descreve a validação por fold (linha 696-700, não
  mais "fixed in advance"), a frase de assimetria 512×10k/30k está em
  main(1).tex:700, as tabelas do RQ4 (main(1).tex:1626-1631 cross-project,
  main(1).tex:1658 within-corpus) têm os números novos, e o Finding 5
  (main(1).tex:1827) e o parágrafo do within-corpus (main(1).tex:1668-1670)
  já refletem a reanálise estatística (CodeBERT bate GPT-OSS com significância
  no within-corpus, g=.638, p_Holm=.0132).
- [ ] **Carta de resposta ao revisor** cobrindo o que continua sem solução
  técnica (decisão consciente, não lacuna): FLACCS/HQCM entram com pesos de
  outro corpus (fora de distribuição); nenhum baseline tipo LoRA-num-7B foi
  adicionado. (Isso é para a submissão, não faz parte do pacote de replicação
  em si.)

## Já validado como OK

- Metodologia (seeds, consenso first-to-two, `num_ctx`/`num_predict`, corte de 30k tokens, taxonomia de 11 classes) bate entre `main (1).tex` e `src/consensus.py` / `src/inference/ollama.py` / `config/evaluation.yaml`.
- `results/` já populado (não são stubs) para RQ1–RQ4.
- Parser canônico (`strict_response_parser.py` v4.1.1) já instalado neste checkout.
- Runner Ollama (`scripts/run_ollama_experiment.py`) agora aceita `--rq`/`--stage` para rodar qualquer RQ/sub-experimento por parâmetro, sem precisar saber o caminho do YAML.
