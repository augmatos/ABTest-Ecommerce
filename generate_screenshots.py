"""
Gera imagens estáticas dos gráficos do teste A/B para o README.
Uso: python generate_screenshots.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'marketing_AB.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(OUT_DIR, exist_ok=True)

COR_T, COR_C = '#4f46e5', '#94a3b8'
LAYOUT = dict(plot_bgcolor='white', paper_bgcolor='white',
              font=dict(family='Arial', size=13), margin=dict(l=50, r=40, t=60, b=40))

np.random.seed(42)


def carregar():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df['converted'] = df['converted'].astype(str).str.lower().map(
        {'true': 1, 'false': 0, '1': 1, '0': 0}).astype(int)
    return df


def stats_base(df):
    g = df.groupby('test_group')['converted'].agg(['sum', 'size'])
    ct, nt = int(g.loc['ad', 'sum']), int(g.loc['ad', 'size'])
    cc, nc = int(g.loc['psa', 'sum']), int(g.loc['psa', 'size'])
    return ct, nt, cc, nc


def gera_conversao(ct, nt, cc, nc):
    pt, pc = ct / nt, cc / nc
    z = abs(stats.norm.ppf(0.025))
    err_t = z * np.sqrt(pt * (1 - pt) / nt)
    err_c = z * np.sqrt(pc * (1 - pc) / nc)
    _, pval = proportions_ztest([ct, cc], [nt, nc])

    fig = go.Figure(go.Bar(
        x=['Controle (psa)', 'Tratamento (ad)'], y=[pc * 100, pt * 100],
        error_y=dict(type='data', array=[err_c * 100, err_t * 100], visible=True, thickness=2),
        marker_color=[COR_C, COR_T],
        text=[f'{pc:.3%}', f'{pt:.3%}'], textposition='outside',
    ))
    fig.update_layout(
        title=f'Taxa de Conversão por Grupo (IC 95%)  |  lift +{(pt-pc)/pc:.1%} · p={pval:.1e}',
        yaxis_title='Conversão (%)', height=420, width=800, showlegend=False, **LAYOUT)
    path = os.path.join(OUT_DIR, 'conversao_grupos.png')
    fig.write_image(path, scale=2)
    print(f'Salvo: {path}')


def gera_bootstrap(ct, nt, cc, nc):
    pt, pc = ct / nt, cc / nc
    boot = np.random.binomial(nt, pt, 10000) / nt - np.random.binomial(nc, pc, 10000) / nc
    ci = np.percentile(boot, [2.5, 97.5])
    fig = go.Figure(go.Histogram(x=boot * 100, nbinsx=60, marker_color=COR_T))
    fig.add_vline(x=0, line_dash='dash', line_color='#ef4444', annotation_text='H₀: diff=0')
    fig.add_vline(x=ci[0] * 100, line_dash='dot', line_color='#22c55e')
    fig.add_vline(x=ci[1] * 100, line_dash='dot', line_color='#22c55e')
    fig.update_layout(
        title=f'Bootstrap da diferença de conversão  |  IC 95%: [{ci[0]:.3%}, {ci[1]:.3%}]',
        xaxis_title='Diferença de conversão (pontos percentuais)',
        height=420, width=900, **LAYOUT)
    path = os.path.join(OUT_DIR, 'bootstrap_diferenca.png')
    fig.write_image(path, scale=2)
    print(f'Salvo: {path}')


def gera_bayes_e_poder(ct, nt, cc, nc):
    pt, pc = ct / nt, cc / nc
    # Posteriores Beta-Binomial
    post_c = np.random.beta(1 + cc, 1 + nc - cc, 100000)
    post_t = np.random.beta(1 + ct, 1 + nt - ct, 100000)
    prob = (post_t > post_c).mean()

    # Curva de poder x tamanho de amostra
    analise = NormalIndPower()
    h = proportion_effectsize(pt, pc)
    ns = np.linspace(500, 20000, 50)
    poderes = [analise.power(effect_size=h, nobs1=n, ratio=1, alpha=0.05) for n in ns]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f'Posteriores da conversão (Bayes)  |  P(ad > psa) = {prob:.1%}',
        'Curva de Poder × Tamanho de Amostra (efeito observado)'))

    fig.add_trace(go.Histogram(x=post_c * 100, name='Controle (psa)', opacity=0.6,
                               marker_color=COR_C, nbinsx=80, histnorm='probability density'), row=1, col=1)
    fig.add_trace(go.Histogram(x=post_t * 100, name='Tratamento (ad)', opacity=0.6,
                               marker_color=COR_T, nbinsx=80, histnorm='probability density'), row=1, col=1)

    fig.add_trace(go.Scatter(x=ns, y=poderes, mode='lines', line=dict(color='#4f46e5', width=2.5),
                             showlegend=False), row=1, col=2)
    fig.add_hline(y=0.8, line_dash='dash', line_color='#ef4444', annotation_text='80% (alvo)', row=1, col=2)

    fig.update_xaxes(title_text='Taxa de conversão (%)', row=1, col=1)
    fig.update_xaxes(title_text='Nº de usuários por grupo', row=1, col=2)
    fig.update_yaxes(title_text='Poder', row=1, col=2)
    fig.update_layout(
        height=450, width=1200, barmode='overlay',
        legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.25),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Arial', size=13),
        margin=dict(l=50, r=40, t=60, b=70))
    path = os.path.join(OUT_DIR, 'bayes_e_poder.png')
    fig.write_image(path, scale=2)
    print(f'Salvo: {path}')


if __name__ == '__main__':
    print('Gerando imagens…')
    df = carregar()
    ct, nt, cc, nc = stats_base(df)
    gera_conversao(ct, nt, cc, nc)
    gera_bootstrap(ct, nt, cc, nc)
    gera_bayes_e_poder(ct, nt, cc, nc)
    print('Concluído! Imagens salvas em images/')
