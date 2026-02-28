# MMM Demo — Marketing Mix Modeling Bayesiano

[![CI](https://github.com/morobr/MMM_DEMO/actions/workflows/ci.yml/badge.svg)](https://github.com/morobr/MMM_DEMO/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![PyMC-Marketing](https://img.shields.io/badge/PyMC--Marketing-latest-orange.svg)](https://github.com/pymc-labs/pymc-marketing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Also available in: [English](README.md)

> **Uma demonstração em nível de produção de Marketing Mix Modeling (MMM) Bayesiano usando PyMC-Marketing.**
> Construído com dados reais do varejo indiano para demonstrar atribuição de canais de ponta a ponta,
> modelagem de adstock/saturação e otimização de orçamento baseada em dados.

---

## O Que Este Demo Apresenta

Este projeto percorre o fluxo completo de MMM — dos dados brutos de transações até
um orçamento de mídia otimizado — usando uma abordagem totalmente Bayesiana:

| Etapa | O Que Você Aprende |
|-------|-------------------|
| **Engenharia de Dados** | Agregar vendas diárias e investimentos mensais em um painel semanal |
| **Agrupamento de Canais** | Tratar multicolinearidade agrupando canais correlacionados |
| **Modelagem Bayesiana** | Seleção de priors, GeometricAdstock, LogisticSaturation |
| **Diagnósticos** | R-hat, ESS, divergências, verificações preditivas a priori/posteriori |
| **Decomposição** | Isolar a contribuição de cada canal no GMV total |
| **Otimização** | Encontrar a alocação de orçamento que maximiza as vendas previstas |

---

## Dataset

**Fonte:** [DT Mart Market Mix Modeling — Kaggle](https://www.kaggle.com/datasets/datatattle/dt-mart-market-mix-modeling)

Dados reais de e-commerce indiano cobrindo um ano de operações (Jul 2015 – Jun 2016):

| Arquivo | Descrição |
|---------|-----------|
| `Sales.csv` | Registros de transações diárias brutas (~1M linhas) |
| `firstfile.csv` | GMV diário agregado por vertical de produto |
| `Secondfile.csv` | Dataset mensal de MMM pré-construído (12 linhas) |
| `MediaInvestment.csv` | Investimento mensal em 9 canais de marketing (valores em Crores INR) |
| `MonthlyNPSscore.csv` | Net Promoter Score mensal (intervalo: 44–60) |
| `SpecialSale.csv` | 44 datas de eventos de vendas especiais em 12 promoções |

Os dados **não são versionados** no repositório. São baixados automaticamente do KaggleHub
na primeira execução via `mmm_demo.data.load_mmm_weekly_data()`.

### Configuração dos Canais

Sete colunas brutas de investimento são agrupadas em quatro canais no modelo para reduzir a multicolinearidade
(correlações par-a-par de até r = 0,99 entre canais correlacionados):

| Canal no Modelo | Colunas Brutas | Participação no Investimento |
|-----------------|----------------|------------------------------|
| **TV** | TV | 5,6% |
| **Sponsorship** | Sponsorship | 46,6% |
| **Digital** | Digital, SEM, Content.Marketing | 16,3% |
| **Online** | Online.marketing, Affiliates | 31,5% |

Rádio e Outros são excluídos (9/12 meses com dados ausentes).

---

## Início Rápido

### Pré-requisitos

- Python 3.12+
- Gerenciador de pacotes [uv](https://docs.astral.sh/uv/)
- Um [token da API do Kaggle](https://www.kaggle.com/docs/api) em `~/.kaggle/kaggle.json`

### Configuração

```bash
# Clonar o repositório
git clone https://github.com/morobr/MMM_DEMO.git
cd MMM_DEMO

# Criar ambiente virtual e instalar todas as dependências
uv sync

# Instalar hooks de pre-commit
uv run pre-commit install

# Verificar instalação
uv run python -c "import pymc_marketing; print('Pronto:', pymc_marketing.__version__)"
```

### Executar os Notebooks

```bash
uv run jupyter lab
```

Abra os notebooks em ordem a partir do diretório `notebooks/`. Cada notebook parte dos
resultados do anterior.

### Executar o Pipeline Completo (não-interativo)

```bash
uv run python scripts/run_pipeline.py
```

Executa todas as etapas do pipeline de ponta a ponta: carregamento dos dados → ajuste do modelo →
diagnósticos → decomposição → otimização → saídas salvas.

### Executar Testes

```bash
uv run pytest -v
```

---

## Notebooks

O demo é estruturado como uma sequência progressiva de notebooks, cada um focado em uma
etapa do fluxo de MMM:

| Notebook | Descrição |
|----------|-----------|
| [`01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) | Visão geral do dataset, validação de schema, análise de correlação, justificativa para seleção de canais |
| [`02_model_fitting.ipynb`](notebooks/02_model_fitting.ipynb) | Construção do painel semanal, verificações preditivas a priori, amostragem MCMC, inspeção do trace |
| [`03_diagnostics.ipynb`](notebooks/03_diagnostics.ipynb) | Validação de convergência (R-hat, ESS, divergências), verificações preditivas a posteriori |
| [`04_contributions.ipynb`](notebooks/04_contributions.ipynb) | Decomposição de canais, curvas de adstock/saturação, atribuição de GMV |
| [`05_optimization.ipynb`](notebooks/05_optimization.ipynb) | Otimização de orçamento com BudgetOptimizer, comparação de cenários, análise de ROI |

> **Dica:** Os notebooks 03–05 dependem de um modelo ajustado salvo em `outputs/models/`.
> Execute o notebook 02 primeiro, ou use o trace pré-ajustado, se disponível.

---

## Estrutura do Projeto

```
MMM_DEMO/
├── README.md
├── README.pt-BR.md                # Esta versão em português
├── CLAUDE.md                      # Especificação do projeto e instruções de IA
├── pyproject.toml                 # Dependências e configuração de ferramentas
├── uv.lock
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       └── ci.yml                 # Lint + format + test a cada push
│
├── notebooks/                     # Sequência progressiva do demo
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_fitting.ipynb
│   ├── 03_diagnostics.ipynb
│   ├── 04_contributions.ipynb
│   └── 05_optimization.ipynb
│
├── scripts/
│   └── run_pipeline.py            # Executor do pipeline de ponta a ponta
│
├── src/
│   └── mmm_demo/
│       ├── __init__.py
│       ├── config.py              # Dataclass ModelConfig — todos os hiperparâmetros
│       ├── data.py                # Carregamento, validação e agregação semanal dos dados
│       ├── model.py               # Construção + ajuste do MMM + amostragem preditiva
│       ├── diagnostics.py         # Verificações de R-hat, ESS, divergências
│       ├── optimization.py        # Wrapper do BudgetOptimizer
│       └── plotting.py            # Todas as funções de visualização
│
├── tests/
│   ├── conftest.py                # Fixtures compartilhadas e ajustes de modelo simulados
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_diagnostics.py
│   ├── test_optimization.py
│   └── test_plotting.py
│
├── data/                          # Baixado automaticamente do KaggleHub (gitignored)
└── outputs/                       # Artefatos gerados (gitignored)
    ├── models/                    # Traces ajustados — mmm_fit_{data}_{desc}.nc
    ├── plots/
    │   ├── diagnostics/           # Plots de trace e preditivo a posteriori
    │   └── contributions/         # Plots de decomposição de canais
    ├── tables/                    # CSVs de resumo
    └── optimization/              # Resultados de cenários de orçamento
```

---

## Arquitetura

O pipeline segue um **fluxo sequencial com portões de validação** — cada etapa deve ser aprovada
antes da próxima começar:

```
Dados Brutos do KaggleHub
        │
        ▼
  [data.py] Carregar e validar schema
        │
        ▼
  [data.py] Construir painel semanal
  (agregar GMV, distribuir investimento,
   atribuir NPS, contar dias de promoção,
   agrupar canais correlacionados)
        │
        ▼
  [config.py] ModelConfig
  (canais, controles, priors,
   adstock_max_lag, parâmetros de amostragem)
        │
        ▼
  [model.py] Construir MMM
  (GeometricAdstock + LogisticSaturation)
        │
        ▼
  [model.py] Verificação Preditiva a Priori
  ── verificar faixas plausíveis de GMV ──
        │
        ▼
  [model.py] Amostragem MCMC
  (4 cadeias × 1000 amostras + 2000 tune)
        │
        ▼
  [diagnostics.py] Portão de Convergência ◄── DEVE PASSAR (R-hat < 1,01, ESS > 400, divergências = 0)
        │
        ▼
  [model.py] Verificação Preditiva a Posteriori
        │
        ▼
  [model.py] Decomposição de Canais
        │
        ▼
  [optimization.py] Otimização de Orçamento
        │
        ▼
  [plotting.py] Salvar Todas as Saídas
        │
        ▼
  outputs/ (modelos, plots, tabelas, otimização)
```

---

## Conceitos de MMM

### Adstock (Efeitos de Carryover)

O investimento em marketing hoje continua a influenciar as vendas por dias ou semanas.
O `GeometricAdstock` modela isso com decaimento exponencial:

```
adstock_t = gasto_t + α × adstock_{t-1}
```

onde `α ∈ (0, 1)` é a taxa de decaimento e `l_max` limita a janela de defasagem (padrão: 4 semanas).

### Saturação (Retornos Decrescentes)

Dobrar o investimento em mídia não dobra o efeito.
O `LogisticSaturation` aplica uma transformação em curva S:

```
saturação(x) = (1 - exp(-λx)) / (1 + exp(-λx))
```

onde `λ` controla a inclinação da curva. λ maior → saturação mais rápida.

### Priors

Todos os priors são calibrados para dados escalonados por MaxAbsScaler (o PyMC-Marketing escala as entradas internamente):

| Parâmetro | Prior | Justificativa |
|-----------|-------|---------------|
| `intercept` | Normal(0, 2) | Baseline não-informativo |
| `adstock_alpha` | Beta(1, 3) | Inclinado para baixo decaimento (carryover curto) |
| `saturation_lam` | Gamma(3, 1) | Saturação moderada esperada |
| `saturation_beta` | HalfNormal(2) | Coeficientes de canal não-negativos |
| `gamma_control` | Normal(0, 2) | Efeitos de controle não-informativos |

### Por Que Bayesiano?

O MMM Bayesiano fornece quantificação completa de incerteza. Em vez de estimativas pontuais para
o ROI dos canais, você obtém distribuições a posteriori — permitindo tomadas de decisão robustas que
levam em conta a incerteza do modelo.

---

## Stack Tecnológica

| Ferramenta | Função |
|------------|--------|
| [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) | Framework MMM (GeometricAdstock, LogisticSaturation, BudgetOptimizer) |
| [PyMC 5.x](https://www.pymc.io/) | Motor Bayesiano (amostragem MCMC via NUTS) |
| [ArviZ](https://python.arviz.org/) | Análise a posteriori, diagnósticos, visualização de trace |
| [pandas](https://pandas.pydata.org/) | Manipulação de dados e engenharia de features |
| [numpy](https://numpy.org/) | Operações numéricas |
| [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) | Visualização |
| [kagglehub](https://github.com/Kaggle/kagglehub) | Download do dataset |
| [uv](https://docs.astral.sh/uv/) | Gerenciador rápido de pacotes e projetos Python |
| [ruff](https://docs.astral.sh/ruff/) | Linting e formatação |
| [pytest](https://pytest.org/) | Testes |

---

## Desenvolvimento

### Comandos Comuns

```bash
# Lint
uv run ruff check .

# Corrigir problemas de lint automaticamente
uv run ruff check --fix .

# Formatar
uv run ruff format .

# Verificação completa (lint + format + testes)
uv run ruff check . && uv run ruff format --check . && uv run pytest

# Executar um módulo de teste específico
uv run pytest tests/test_data.py -v
```

### Filosofia de Testes

- **Testes unitários** para todas as funções públicas em cada módulo
- **Fixtures sintéticas** imitando o schema do DT Mart (sem dados reais nos testes)
- **MCMC simulado** — ajustes de modelo são sempre mockados para manter a suíte de testes rápida
- CI executa a cada push via GitHub Actions

---

## Melhorias Sugeridas (Roadmap)

Este é um projeto de demonstração. Lacunas conhecidas e próximos passos lógicos:

- [ ] Decaimento de adstock variante no tempo (carryover não-estacionário)
- [ ] Validação holdout (divisão treino/teste para avaliação fora da amostra)
- [ ] Múltiplos cenários de orçamento (conservador, base, agressivo)
- [ ] Priors hierárquicos entre canais
- [ ] Exportar tabelas de contribuição para `outputs/tables/`

---

## Limitações Conhecidas

- **Dataset pequeno:** Apenas ~52 observações semanais. As posterioris são amplas e os priors têm
  influência significativa. Os resultados devem ser interpretados com cautela.
- **Mercado único:** Apenas um mercado de varejo; sem segmentação regional/geográfica.
- **Sem regressores externos:** Atividade de concorrentes, mudanças de preço e fatores
  macroeconômicos não são modelados.
- **Distribuição mensal → semanal:** O investimento em mídia é distribuído proporcionalmente entre
  as semanas, o que é uma simplificação do cronograma real.

---

## Referências

- [PyMC-Marketing GitHub](https://github.com/pymc-labs/pymc-marketing)
- [Marketing Mix Modeling — Um Guia Completo (PyMC Labs)](https://www.pymc-labs.com/blog-posts/marketing-mix-modeling-a-complete-guide)
- [Dataset DT Mart no Kaggle](https://www.kaggle.com/datasets/datatattle/dt-mart-market-mix-modeling)
- [Documentação ArviZ](https://python.arviz.org/)

---

## Licença

Licença MIT. Veja [LICENSE](LICENSE) para detalhes.
