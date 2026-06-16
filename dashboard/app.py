"""
Dashboard Interativo — Análise de Teste A/B (Conversão)
Autor: Augusto Matos

Como rodar:
    pip install -r requirements.txt
    python dashboard/app.py

Pré-requisito: marketing_AB.csv na pasta data/.
Acesse: http://localhost:8050
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

# ── Carregamento e cálculos base (uma vez) ───────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'marketing_AB.csv')

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
df['converted'] = df['converted'].astype(str).str.lower().map(
    {'true': 1, 'false': 0, '1': 1, '0': 0}).astype(int)

_g = df.groupby('test_group')['converted'].agg(['sum', 'size'])
CONV_T, N_T = int(_g.loc['ad', 'sum']), int(_g.loc['ad', 'size'])
CONV_C, N_C = int(_g.loc['psa', 'sum']), int(_g.loc['psa', 'size'])
P_T, P_C = CONV_T / N_T, CONV_C / N_C
LIFT_ABS = P_T - P_C
LIFT_REL = LIFT_ABS / P_C

Z_STAT, P_VALUE = proportions_ztest([CONV_T, CONV_C], [N_T, N_C], alternative='two-sided')

# Bootstrap (precomputado)
np.random.seed(42)
_N_BOOT = 10_000
_boot_diff = (np.random.binomial(N_T, P_T, _N_BOOT) / N_T -
              np.random.binomial(N_C, P_C, _N_BOOT) / N_C)

COR_T, COR_C = '#4f46e5', '#94a3b8'

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Teste A/B — Conversão'

CARD_STYLE = {'borderRadius': '10px', 'padding': '16px 20px', 'background': '#fff',
              'border': '1.5px solid #e5e7eb', 'textAlign': 'center'}

def kpi_card(titulo, valor, cor='#4f46e5'):
    return html.Div([
        html.P(titulo, style={'fontSize': '12px', 'color': '#6b7280', 'margin': '0'}),
        html.H4(valor, style={'color': cor, 'margin': '4px 0 0', 'fontSize': '22px', 'fontWeight': '700'})
    ], style=CARD_STYLE)


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H3('🧪 Teste A/B — Taxa de Conversão',
                        style={'color': '#1a1a2e', 'fontWeight': '700', 'margin': '0'})),
        dbc.Col(html.P('Dataset: Marketing A/B | ad (tratamento) vs psa (controle)',
                       style={'color': '#6b7280', 'textAlign': 'right', 'margin': '8px 0 0'}))
    ], align='center', className='mb-3 mt-3'),

    dbc.Row([
        dbc.Col([
            html.Label('Nível de confiança:', style={'fontWeight': '600', 'fontSize': '13px'}),
            dcc.Dropdown(
                id='nivel-confianca',
                options=[{'label': '90%', 'value': 0.90},
                         {'label': '95%', 'value': 0.95},
                         {'label': '99%', 'value': 0.99}],
                value=0.95, clearable=False, style={'fontSize': '13px'}),
        ], md=3),
    ], className='mb-3'),

    dbc.Row(id='kpi-cards', className='mb-3 g-2'),

    dbc.Row([
        dbc.Col(dcc.Graph(id='grafico-conversao'), md=6),
        dbc.Col(dcc.Graph(id='grafico-bootstrap'), md=6),
    ], className='mb-3'),

    dbc.Row([dbc.Col(html.Div(id='veredito'), md=12)], className='mb-4'),

], fluid=True, style={'backgroundColor': '#f9fafb', 'minHeight': '100vh', 'padding': '0 24px'})


@app.callback(
    Output('kpi-cards', 'children'),
    Output('grafico-conversao', 'figure'),
    Output('grafico-bootstrap', 'figure'),
    Output('veredito', 'children'),
    Input('nivel-confianca', 'value'),
)
def atualizar(nivel):
    alpha = 1 - nivel

    # IC para a diferença de proporções no nível escolhido
    ci_low, ci_upp = confint_proportions_2indep(
        CONV_T, N_T, CONV_C, N_C, compare='diff', method='wald', alpha=alpha)
    # Margem de erro de cada proporção (para as barras de erro), no nível escolhido
    z_critico = abs(stats.norm.ppf(alpha / 2))
    def margem_erro(conv, n):
        p = conv / n
        se = np.sqrt(p * (1 - p) / n)
        return z_critico * se
    err_t = margem_erro(CONV_T, N_T)
    err_c = margem_erro(CONV_C, N_C)

    significativo = (ci_low > 0) or (ci_upp < 0)

    cards = [
        dbc.Col(kpi_card('Conversão Controle', f'{P_C:.3%}', COR_C), md=2),
        dbc.Col(kpi_card('Conversão Tratamento', f'{P_T:.3%}', COR_T), md=2),
        dbc.Col(kpi_card('Lift Relativo', f'{LIFT_REL:+.1%}', '#22c55e'), md=2),
        dbc.Col(kpi_card('Lift Absoluto', f'{LIFT_ABS:+.3%}', '#22c55e'), md=2),
        dbc.Col(kpi_card('p-value', f'{P_VALUE:.2e}', '#ef4444'), md=2),
        dbc.Col(kpi_card(f'IC {nivel:.0%} da diferença',
                         f'[{ci_low:.3%}, {ci_upp:.3%}]', '#1a1a2e'), md=2),
    ]

    # Gráfico de conversão com barras de erro
    fig_conv = go.Figure(go.Bar(
        x=['Controle (psa)', 'Tratamento (ad)'],
        y=[P_C * 100, P_T * 100],
        error_y=dict(type='data', array=[err_c * 100, err_t * 100], visible=True),
        marker_color=[COR_C, COR_T],
        text=[f'{P_C:.3%}', f'{P_T:.3%}'], textposition='outside',
    ))
    fig_conv.update_layout(title=f'Taxa de Conversão por Grupo (IC {nivel:.0%})',
                           yaxis_title='Conversão (%)', height=400, template='plotly_white',
                           margin=dict(l=40, r=20, t=50, b=30))

    # Bootstrap da diferença
    ci_boot = np.percentile(_boot_diff, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    fig_boot = go.Figure(go.Histogram(x=_boot_diff * 100, nbinsx=60, marker_color=COR_T))
    fig_boot.add_vline(x=0, line_dash='dash', line_color='#ef4444')
    fig_boot.add_vline(x=ci_boot[0] * 100, line_dash='dot', line_color='#22c55e')
    fig_boot.add_vline(x=ci_boot[1] * 100, line_dash='dot', line_color='#22c55e')
    fig_boot.update_layout(title=f'Bootstrap da diferença de conversão (IC {nivel:.0%})',
                           xaxis_title='Diferença (pontos percentuais)', height=400,
                           template='plotly_white', margin=dict(l=40, r=20, t=50, b=30))

    # Veredito textual
    cor_v = '#16a34a' if significativo else '#dc2626'
    txt = ('✅ Diferença ESTATISTICAMENTE SIGNIFICATIVA' if significativo
           else '❌ Diferença NÃO significativa')
    veredito = html.Div([
        html.H5(txt, style={'color': cor_v, 'fontWeight': '700', 'margin': '0 0 6px'}),
        html.P(f'A {nivel:.0%} de confiança, o intervalo da diferença '
               f'{"exclui" if significativo else "inclui"} o zero. '
               f'Z = {Z_STAT:.2f}, p = {P_VALUE:.2e}. '
               f'Lift relativo de {LIFT_REL:+.1%} (~{LIFT_ABS * N_T:,.0f} conversões incrementais no tratamento).',
               style={'color': '#374151', 'margin': '0', 'fontSize': '14px'}),
    ], style={**CARD_STYLE, 'textAlign': 'left'})

    return cards, fig_conv, fig_boot, veredito


if __name__ == '__main__':
    app.run(debug=True)
