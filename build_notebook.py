"""Gera o notebook 01_analise_ab.ipynb via nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

# ── Título ────────────────────────────────────────────────────────────────
md("""# 🧪 Análise de Teste A/B — Conversão em E-commerce

Análise estatística completa de um teste A/B de marketing, decidindo se a variação testada
gera um aumento **estatisticamente significativo** na taxa de conversão — e quantificando a
magnitude e a confiança desse efeito.

**Dataset:** [Marketing A/B Testing](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing) —
~588 mil usuários divididos em dois grupos:
- **`ad`** (tratamento): usuários expostos aos anúncios da campanha
- **`psa`** (controle): usuários expostos a um aviso institucional (placebo)

**Roteiro estatístico:**
1. EDA e checagem de balanceamento dos grupos
2. **Teste Z de proporções** sobre a taxa de conversão (métrica primária)
3. **Intervalo de confiança** para a diferença de proporções
4. **Tamanho de efeito** (Cohen's h) — significância prática, não só estatística
5. **Análise de poder** e **tamanho de amostra** necessário (MDE)
6. **Bootstrap** — validação não-paramétrica do resultado
7. **Abordagem Bayesiana** (Beta-Binomial) — P(tratamento > controle)
8. **Teste t / Mann-Whitney** sobre uma métrica contínua (exposição)
9. Conclusão e recomendação de negócio
""")

# ── Setup ─────────────────────────────────────────────────────────────────
code("""import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import stats
from statsmodels.stats.proportion import (
    proportions_ztest, confint_proportions_2indep, proportion_effectsize
)
from statsmodels.stats.power import NormalIndPower

np.random.seed(42)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
ALPHA = 0.05  # nível de significância (confiança de 95%)
""")

md("## 1. Carregamento e preparação")

code("""df = pd.read_csv('../data/marketing_AB.csv')
# Padroniza nomes de colunas (o CSV vem com espaços e uma coluna de índice)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
df = df.rename(columns={'unnamed:_0': 'idx'})
print('Colunas:', list(df.columns))
df.head()
""")

code("""# 'converted' pode vir como bool ou string — normalizamos para 0/1
df['converted'] = df['converted'].astype(str).str.lower().map(
    {'true': 1, 'false': 0, '1': 1, '0': 0}
).astype(int)

# Renomeia os grupos para leitura clara
df['grupo'] = df['test_group'].map({'ad': 'Tratamento (ad)', 'psa': 'Controle (psa)'})

print(f'Total de usuários: {len(df):,}')
print(df['grupo'].value_counts())
""")

md("## 2. EDA e balanceamento dos grupos")

code("""resumo = df.groupby('grupo').agg(
    usuarios=('converted', 'size'),
    conversoes=('converted', 'sum'),
    taxa_conversao=('converted', 'mean'),
).reset_index()
resumo['taxa_conversao_%'] = (resumo['taxa_conversao'] * 100).round(3)
resumo
""")

code("""# Os grupos não precisam ter o mesmo tamanho, mas é bom saber o desbalanceamento:
# um grupo placebo (psa) costuma ser bem menor. O teste Z de 2 amostras lida com
# tamanhos diferentes naturalmente.
prop_grupos = df['grupo'].value_counts(normalize=True) * 100
fig = go.Figure(go.Bar(x=prop_grupos.index, y=prop_grupos.values,
                        marker_color=['#4f46e5', '#94a3b8'],
                        text=[f'{v:.1f}%' for v in prop_grupos.values], textposition='outside'))
fig.update_layout(title='Distribuição dos usuários entre os grupos', yaxis_title='% dos usuários',
                  height=350, template='plotly_white')
fig.show()
""")

md("""## 3. Métrica primária — Taxa de Conversão

### Por que o Teste Z de proporções?

A conversão é uma variável **binária** (converteu / não converteu) e queremos comparar a
**proporção de sucesso entre dois grupos independentes**, com amostras grandes (n na casa
das centenas de milhares). Nesse cenário, a distribuição amostral da diferença de proporções
é aproximadamente normal (Teorema Central do Limite), o que torna o **teste Z de duas proporções**
o método adequado — mais apropriado que um teste t (usado para médias de variáveis contínuas)
ou um qui-quadrado (equivalente, mas que não entrega diretamente a direção/IC da diferença).

- **H₀:** a taxa de conversão é igual nos dois grupos (p_tratamento = p_controle)
- **H₁:** as taxas são diferentes (teste bicaudal)
- **α = 0,05** (95% de confiança)""")

code("""conv = df.groupby('grupo')['converted'].agg(['sum', 'size'])
conv_treat, n_treat = int(conv.loc['Tratamento (ad)', 'sum']), int(conv.loc['Tratamento (ad)', 'size'])
conv_ctrl,  n_ctrl  = int(conv.loc['Controle (psa)', 'sum']),  int(conv.loc['Controle (psa)', 'size'])

p_treat = conv_treat / n_treat
p_ctrl  = conv_ctrl / n_ctrl
lift_abs = p_treat - p_ctrl
lift_rel = lift_abs / p_ctrl

print(f'Controle (psa):     {p_ctrl:.4%}  ({conv_ctrl:,} / {n_ctrl:,})')
print(f'Tratamento (ad):    {p_treat:.4%}  ({conv_treat:,} / {n_treat:,})')
print(f'Lift absoluto:      {lift_abs:.4%} (p.p.)')
print(f'Lift relativo:      {lift_rel:.2%}')
""")

code("""# Teste Z de proporções (bicaudal)
count = np.array([conv_treat, conv_ctrl])
nobs  = np.array([n_treat, n_ctrl])
z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')

print(f'Estatística Z: {z_stat:.4f}')
print(f'p-value:       {p_value:.3e}')
print(f'Significativo a α={ALPHA}? {\"SIM\" if p_value < ALPHA else \"NÃO\"}')
""")

md("## 4. Intervalo de confiança para a diferença de proporções")

code("""ci_low, ci_upp = confint_proportions_2indep(
    conv_treat, n_treat, conv_ctrl, n_ctrl, compare='diff', method='wald', alpha=ALPHA
)
print(f'Diferença de conversão (tratamento - controle): {lift_abs:.4%}')
print(f'IC 95%: [{ci_low:.4%}, {ci_upp:.4%}]')
print('O IC não contém zero -> diferença significativa.' if (ci_low > 0 or ci_upp < 0)
      else 'O IC contém zero -> não significativa.')
""")

md("""## 5. Tamanho de efeito (Cohen's h)

O p-value diz **se** há diferença, mas não **o quão grande** ela é — com n gigante, diferenças
ínfimas viram "significativas". O **Cohen's h** mede a magnitude do efeito entre duas proporções,
independente do tamanho da amostra. Referência: ~0,2 pequeno, ~0,5 médio, ~0,8 grande.""")

code("""h = proportion_effectsize(p_treat, p_ctrl)
magnitude = ('grande' if abs(h) >= 0.8 else 'médio' if abs(h) >= 0.5
             else 'pequeno' if abs(h) >= 0.2 else 'muito pequeno')
print(f\"Cohen's h: {h:.4f}  ->  efeito {magnitude}\")
print(f'Lift relativo (referência de negócio): {lift_rel:.1%}')
""")

md("""> ⚠️ **Cuidado ao interpretar Cohen's h em eventos raros.** Aqui o `h` é classificado como
> "muito pequeno", mas o **lift relativo é de +43%** — um ganho enorme em conversão. Isso não é
> contradição: o Cohen's h opera na escala arco-seno e **comprime diferenças entre proporções
> próximas de zero**. Quando a conversão base é baixa (~1,8%), a métrica mais informativa para o
> negócio é o **lift relativo** e, sobretudo, o **lift absoluto × volume de usuários** (impacto
> em conversões adicionais), não o `h` isolado. Lição: nenhuma métrica de efeito deve ser lida
> fora do contexto da taxa-base.""")

code("""# Impacto de negócio: conversões incrementais atribuíveis ao tratamento
# (aplicando o lift absoluto à base exposta ao tratamento)
conversoes_incrementais = lift_abs * n_treat
print(f'Lift absoluto: {lift_abs:.4%} por usuário')
print(f'Usuários no tratamento: {n_treat:,}')
print(f'Conversões incrementais estimadas: ~{conversoes_incrementais:,.0f}')
print(f'(intervalo via IC 95%: {ci_low * n_treat:,.0f} a {ci_upp * n_treat:,.0f})')
""")

md("""## 6. Análise de poder e tamanho de amostra

- **Poder (post-hoc):** dada a amostra e o efeito observados, qual a probabilidade de detectarmos
  o efeito se ele realmente existe? (alvo usual: ≥ 80%)
- **Sample size (planejamento):** quantos usuários por grupo seriam necessários para detectar um
  efeito mínimo de interesse (MDE) — útil para dimensionar testes futuros.""")

code("""analise_poder = NormalIndPower()

poder = analise_poder.power(effect_size=h, nobs1=n_treat,
                            ratio=n_ctrl / n_treat, alpha=ALPHA, alternative='two-sided')
print(f'Poder estatístico (post-hoc): {poder:.2%}')

# Sample size por grupo para detectar um MDE de +1 p.p. sobre o controle, com 80% de poder
mde = 0.01
h_mde = proportion_effectsize(p_ctrl + mde, p_ctrl)
n_necessario = analise_poder.solve_power(effect_size=h_mde, alpha=ALPHA, power=0.80,
                                         ratio=1, alternative='two-sided')
print(f'Para detectar +{mde:.0%} p.p. (80% poder): {np.ceil(n_necessario):,.0f} usuários por grupo')
""")

md("""## 7. Validação por Bootstrap

Reamostramos a diferença de conversão milhares de vezes para construir empiricamente sua
distribuição — sem depender da suposição de normalidade. Se o IC bootstrap também excluir o
zero, reforçamos a conclusão do teste paramétrico.""")

code("""N_BOOT = 10_000
# Reamostragem da proporção de cada grupo (Monte Carlo sobre a Bernoulli estimada)
boot_treat = np.random.binomial(n_treat, p_treat, N_BOOT) / n_treat
boot_ctrl  = np.random.binomial(n_ctrl,  p_ctrl,  N_BOOT) / n_ctrl
boot_diff  = boot_treat - boot_ctrl

ic_boot = np.percentile(boot_diff, [2.5, 97.5])
print(f'Diferença média (bootstrap): {boot_diff.mean():.4%}')
print(f'IC 95% bootstrap: [{ic_boot[0]:.4%}, {ic_boot[1]:.4%}]')
print(f'P(diferença > 0): {(boot_diff > 0).mean():.2%}')
""")

code("""fig = go.Figure(go.Histogram(x=boot_diff * 100, nbinsx=60, marker_color='#4f46e5'))
fig.add_vline(x=0, line_dash='dash', line_color='#ef4444', annotation_text='H₀: diff = 0')
fig.add_vline(x=ic_boot[0] * 100, line_dash='dot', line_color='#22c55e')
fig.add_vline(x=ic_boot[1] * 100, line_dash='dot', line_color='#22c55e')
fig.update_layout(title='Distribuição bootstrap da diferença de conversão (p.p.)',
                  xaxis_title='Diferença de conversão (pontos percentuais)',
                  height=400, template='plotly_white')
fig.show()
""")

md("""## 8. Abordagem Bayesiana (Beta-Binomial)

Além do veredito binário do teste de hipótese, a visão bayesiana responde diretamente à pergunta
do negócio: **"qual a probabilidade de o tratamento ser melhor que o controle?"**. Usamos um prior
Beta(1,1) (uniforme) e atualizamos com os dados, obtendo a distribuição posterior da conversão de
cada grupo.""")

code("""N_SAMPLES = 100_000
post_ctrl  = np.random.beta(1 + conv_ctrl,  1 + n_ctrl  - conv_ctrl,  N_SAMPLES)
post_treat = np.random.beta(1 + conv_treat, 1 + n_treat - conv_treat, N_SAMPLES)

prob_melhor = (post_treat > post_ctrl).mean()
uplift_esperado = (post_treat - post_ctrl).mean()
print(f'P(Tratamento > Controle): {prob_melhor:.2%}')
print(f'Uplift esperado (posterior): {uplift_esperado:.4%}')
""")

code("""fig = go.Figure()
fig.add_trace(go.Histogram(x=post_ctrl * 100, name='Controle (psa)', opacity=0.6,
                           marker_color='#94a3b8', nbinsx=80, histnorm='probability density'))
fig.add_trace(go.Histogram(x=post_treat * 100, name='Tratamento (ad)', opacity=0.6,
                           marker_color='#4f46e5', nbinsx=80, histnorm='probability density'))
fig.update_layout(title='Distribuições posteriores da taxa de conversão',
                  xaxis_title='Taxa de conversão (%)', barmode='overlay',
                  height=400, template='plotly_white')
fig.show()
""")

md("""## 9. Métrica secundária — Exposição (variável contínua)

Para demonstrar o teste sobre uma **variável contínua**, analisamos o nº de anúncios vistos
(`total_ads`): quem converteu foi exposto a mais anúncios? Como contagens costumam ser muito
assimétricas, comparamos o **teste t de Welch** (médias) com o **Mann-Whitney** (não-paramétrico,
sobre as medianas/distribuições) — boa prática quando a normalidade é duvidosa.""")

code("""ads_conv = df.loc[df['converted'] == 1, 'total_ads']
ads_nao  = df.loc[df['converted'] == 0, 'total_ads']

t_stat, t_p = stats.ttest_ind(ads_conv, ads_nao, equal_var=False)   # Welch
u_stat, u_p = stats.mannwhitneyu(ads_conv, ads_nao, alternative='two-sided')

# Cohen's d (tamanho de efeito para médias)
def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp

print(f'Média de anúncios — converteram:    {ads_conv.mean():.1f} (mediana {ads_conv.median():.0f})')
print(f'Média de anúncios — não converteram: {ads_nao.mean():.1f} (mediana {ads_nao.median():.0f})')
print(f'\\nTeste t de Welch:  t={t_stat:.2f}, p={t_p:.3e}')
print(f'Mann-Whitney U:    U={u_stat:.3e}, p={u_p:.3e}')
print(f\"Cohen's d:         {cohens_d(ads_conv, ads_nao):.3f}\")
""")

md("""## 10. Conclusões e recomendação

**Resultado do teste A/B (conversão):**
- A diferença na taxa de conversão entre tratamento e controle é avaliada pelo teste Z de
  proporções, pelo IC da diferença, pelo bootstrap e pela probabilidade bayesiana — todos
  apontando na mesma direção, o que dá robustez à decisão.
- O **tamanho de efeito (Cohen's h)** contextualiza a magnitude: com amostras enormes, é
  essencial separar significância **estatística** de significância **prática/de negócio**.
- A **análise de poder** confirma que a amostra é mais que suficiente para detectar o efeito,
  e o cálculo de sample size serve de referência para dimensionar testes futuros.

**Métrica secundária (exposição):**
- A concordância entre teste t de Welch e Mann-Whitney mostra que a conclusão não depende da
  suposição de normalidade.

**Recomendação:** decidir o rollout da variação com base no efeito observado **e** no seu impacto
prático esperado (uplift × volume), não apenas no p-value.

### Próximos passos
- Testar segmentação do efeito por dia/hora de maior exposição (`most_ads_day`, `most_ads_hour`)
- Monitorar o efeito ao longo do tempo (novelty effect)
- Análise sequencial / correção para múltiplas comparações se vários KPIs forem testados
""")

nb['cells'] = cells
nbf.write(nb, 'notebooks/01_analise_ab.ipynb')
print('Notebook gerado com sucesso!')
