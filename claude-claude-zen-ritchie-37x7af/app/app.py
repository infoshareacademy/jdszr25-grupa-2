"""Radar Upadłości — aplikacja demonstracyjna (Streamlit).

Uruchomienie:  streamlit run app.py
Wymaga pliku models/bundle.joblib (tworzy go train_models.py).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from feature_info import CATEGORIES, CATEGORY_ICONS, describe

BUNDLE_PATH = Path(__file__).parent / 'models' / 'bundle.joblib'

# Nowsze Streamlity używają width='stretch', starsze use_container_width=True
_ver = tuple(int(v) for v in st.__version__.split('.')[:2])
STRETCH = {'width': 'stretch'} if _ver >= (1, 46) else {'use_container_width': True}

RED, AMBER, GREEN, BLUE, VIOLET, GREY = '#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#a855f7', '#94a3b8'

PROFILES = {
    'bank': dict(
        icon='🏦', name='Bank — kredyty',
        short='Wyłapujemy ~99% przyszłych bankrutów. Alarm = wniosek idzie do analityka, a nie automatyczna odmowa.',
        alarm='DO WERYFIKACJI', alarm_short='WERYFIKACJA', ok='AUTOMATYCZNA AKCEPTACJA', color=BLUE),
    'balanced': dict(
        icon='⚖️', name='Monitoring / audyt',
        short='Najlepszy kompromis między wykrywalnością a liczbą fałszywych alarmów (maks. F1).',
        alarm='ALARM', alarm_short='ALARM', ok='BEZ ZASTRZEŻEŃ', color=VIOLET),
    'fund': dict(
        icon='📉', name='Fundusz — krótka sprzedaż',
        short='Sygnał tylko wtedy, gdy model jest naprawdę pewny — liczy się trafność każdej pozycji.',
        alarm='SYGNAŁ SHORT', alarm_short='SHORT', ok='BRAK SYGNAŁU', color=AMBER),
}

st.set_page_config(page_title='Radar Upadłości', page_icon='🛡️', layout='wide')

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {padding: 1.1rem 1.4rem; border-radius: 16px;
       background: linear-gradient(120deg, rgba(99,102,241,.25), rgba(14,165,233,.12));
       border: 1px solid rgba(148,163,184,.25); margin-bottom: 1rem;}
.hero h1 {font-size: 1.9rem; margin: 0;}
.hero p {margin: .25rem 0 0 0; opacity: .85;}
.verdict {border-radius: 16px; padding: 1rem 1.2rem; color: white; margin-bottom: .6rem;}
.verdict .big {font-size: 1.6rem; font-weight: 800; letter-spacing: .5px;}
.verdict .small {opacity: .92; font-size: .95rem;}
.hcard {border-radius: 12px; padding: .55rem .4rem; text-align: center;
        border: 1px solid rgba(148,163,184,.25); background: rgba(148,163,184,.06);}
.hcard .h {font-size: .8rem; opacity: .75;}
.hcard .v {font-size: 1.35rem; font-weight: 700;}
.hcard .s {font-size: .75rem; font-weight: 600;}
.pill {display:inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .78rem;
       font-weight: 600; background: rgba(148,163,184,.18);}
.demo {background: #b45309; color: white; padding: .5rem .9rem; border-radius: 10px; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- dane i model

@st.cache_resource
def load_bundle(path):
    b = joblib.load(path)
    for h in b['horizons'].values():
        booster = xgb.Booster()
        booster.load_model(bytearray(h['booster']))
        h['model'] = booster
        h['q'] = np.asarray(h['score_quantiles'])
    return b


if not BUNDLE_PATH.exists():
    st.error('Brak pliku z modelami `models/bundle.joblib`. Uruchom najpierw: '
             '`python train_models.py --data-dir ŚCIEŻKA_DO_DANYCH`.')
    st.stop()

B = load_bundle(str(BUNDLE_PATH))
FEATS = B['features']
STATS = B['feature_stats']
HORIZONS = sorted(B['horizons'])
INFO = {c: describe(c) for c in FEATS}


def plot(fig, height=None):
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=10, r=10, t=40, b=30), font=dict(size=13),
                      legend=dict(orientation='h', y=-0.32, x=0))
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, config={'displayModeBar': False}, **STRETCH)


def fmt_idx(p):
    """Indeks ryzyka; w samej górze rozkładu pokazujemy miejsce po przecinku (99.5, a nie 100)."""
    return f'{p:.1f}' if 99 <= p < 100 else f'{p:.0f}'


def h_label(k):
    return f'{k} rok' if k == 1 else (f'{k} lata' if k < 5 else f'{k} lat')


def calibrated_pd(score, h):
    odds = score / np.clip(1 - score, 1e-9, None) * B['horizons'][h]['odds_factor']
    return odds / (1 + odds)


def percentile(score, h):
    """Jaki % firm z danych testowych ma NIŻSZY wynik ryzyka."""
    return np.searchsorted(B['horizons'][h]['q'], score) / 10


def score_frame(X):
    """Wynik dla wielu firm i wszystkich horyzontów."""
    X = X[FEATS].astype(float)
    dm = xgb.DMatrix(X)
    out = {}
    for h in HORIZONS:
        s = B['horizons'][h]['model'].predict(dm)
        out[h] = s
    return out


def contributions(x, h):
    dm = xgb.DMatrix(pd.DataFrame([x])[FEATS].astype(float))
    c = B['horizons'][h]['model'].predict(dm, pred_contribs=True)[0]
    return pd.Series(c[:-1], index=FEATS)


# Kierunek ryzyka liczony z danych: czy zdrowe firmy mają wyższą wartość niż bankruci
DIRECTION = {c: (1 if STATS[c]['healthy_median'] >= STATS[c]['bankrupt_median'] else -1) for c in FEATS}


def slider_bounds(c):
    g = STATS[c]['grid']
    lo, hi = g[1], g[99]
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    return float(lo), float(hi)


def clamp(c, v):
    lo, hi = slider_bounds(c)
    return float(min(max(v, lo), hi))


def value_at_risk_level(c, risk):
    """risk=0 -> wartość typowa dla bardzo zdrowej firmy, risk=1 -> dla bardzo zagrożonej."""
    q = (1 - risk) if DIRECTION[c] == 1 else risk
    q = min(max(q, 0.03), 0.97)
    return float(np.interp(q * 100, np.arange(101), STATS[c]['grid']))


def build_presets():
    def by_category(levels, default):
        return {c: value_at_risk_level(c, levels.get(INFO[c]['category'], default)) for c in FEATS}

    presets = {
        '🟢 Stabilna, zdrowa firma': by_category({}, 0.15),
        '🟡 Przeciętna firma': by_category({}, 0.5),
        '🟠 Rentowna, ale bez gotówki': by_category({'Płynność': 0.95, 'Rentowność': 0.2}, 0.35),
        '🟣 Szybki wzrost na kredyt': by_category({'Zadłużenie': 0.92, 'Wzrost': 0.1, 'Płynność': 0.7}, 0.45),
        '🔴 Firma w kryzysie': by_category({}, 0.93),
        '📌 Typowy bankrut (mediana firm przed upadłością)': {c: STATS[c]['bankrupt_median'] for c in FEATS},
    }
    for name, ex in B.get('examples', {}).items():
        presets[f'📁 {name} ({ex["year"]})'] = ex['values']
    return {n: {c: clamp(c, v) for c, v in vals.items()} for n, vals in presets.items()}


PRESETS = build_presets()


def apply_preset():
    vals = PRESETS.get(st.session_state['preset'])
    if vals:
        for c, v in vals.items():
            st.session_state[f'f_{c}'] = v


if 'initialized' not in st.session_state:
    st.session_state['preset'] = '🟠 Rentowna, ale bez gotówki'
    apply_preset()
    st.session_state['initialized'] = True


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown('## 🛡️ Radar Upadłości')
    st.caption('Wczesne ostrzeganie o bankructwie firm na 1–6 lat do przodu')
    if B.get('demo'):
        st.markdown('<div class="demo">⚠️ TRYB DEMO — modele wytrenowane na danych syntetycznych</div>',
                    unsafe_allow_html=True)
    st.markdown('### Dla kogo liczymy?')
    profile = st.radio('Profil użytkownika', list(PROFILES), key='profile',
                       format_func=lambda p: f'{PROFILES[p]["icon"]} {PROFILES[p]["name"]}',
                       label_visibility='collapsed')
    P = PROFILES[profile]
    st.info(P['short'])
    main_h = st.select_slider('Horyzont szczegółowej analizy', options=HORIZONS,
                              format_func=h_label, key='main_h')
    m = B['horizons'][main_h]['profiles'][profile]
    st.markdown(f'**Jakość profilu na teście ({h_label(main_h)})**')
    c1, c2 = st.columns(2)
    c1.metric('Wykrywalność', f'{m["recall"]:.0%}', help='Recall: jaki % przyszłych bankrutów dostaje alarm.')
    c2.metric('Trafność', f'{m["precision"]:.1%}', help='Precision: jaki % alarmów to faktyczne bankructwa.')
    c1.metric('ROC-AUC', f'{B["horizons"][main_h]["roc_auc"]:.3f}')
    c2.metric('Alarmów', f'{m["alert_rate"]:.1%}', help='Jaki % wszystkich firm dostaje alarm.')
    st.caption(f'Model: XGBoost · {len(FEATS)} cech · dane V4FinBench · '
               f'zbudowano {B["created"].replace("T", " ")}')


st.markdown(f"""
<div class="hero">
  <h1>🛡️ Radar Upadłości</h1>
  <p>Na podstawie {len(FEATS)} wskaźników ze sprawozdania finansowego oceniamy ryzyko bankructwa firmy
  w horyzoncie od 1 do {max(HORIZONS)} lat. Profil: <b>{P['icon']} {P['name']}</b>.</p>
</div>
""", unsafe_allow_html=True)

tab_firm, tab_port, tab_value, tab_quality, tab_about = st.tabs(
    ['🔎 Ocena firmy', '📂 Portfel firm', '💼 Wartość biznesowa', '🧪 Jakość modelu', 'ℹ️ Jak to działa'])


# ---------------------------------------------------------------- 1. ocena firmy

def verdict_html(is_alarm, title, subtitle):
    bg = f'linear-gradient(120deg, {RED}, #b91c1c)' if is_alarm else f'linear-gradient(120deg, {GREEN}, #15803d)'
    return (f'<div class="verdict" style="background:{bg}"><div class="small">{subtitle}</div>'
            f'<div class="big">{"🚨" if is_alarm else "✅"} {title}</div></div>')


def risk_word(pct):
    if pct >= 95:
        return 'bardzo wysokie', RED
    if pct >= 80:
        return 'podwyższone', AMBER
    if pct >= 50:
        return 'umiarkowane', '#eab308'
    return 'niskie', GREEN


with tab_firm:
    left, right = st.columns([5, 7], gap='large')

    with left:
        st.selectbox('Wczytaj przykładową firmę', list(PRESETS), key='preset', on_change=apply_preset,
                     help='Archetypy zbudowane z rozkładów prawdziwych danych oraz prawdziwe (anonimowe) '
                          'firmy ze zbioru testowego.')
        st.caption('Przesuwaj suwaki — wynik przelicza się na żywo. Najedź na ⓘ, żeby zobaczyć objaśnienie.')
        x = {}
        cats = [c for c in CATEGORIES if any(INFO[f]['category'] == c for f in FEATS)]
        for cat in cats:
            with st.container(border=True):
                st.markdown(f'**{CATEGORY_ICONS[cat]} {cat}**')
                for c in [f for f in FEATS if INFO[f]['category'] == cat]:
                    lo, hi = slider_bounds(c)
                    step = float(10 ** np.floor(np.log10((hi - lo) / 200))) if hi > lo else 0.01
                    good = {1: '↑ wyżej = bezpieczniej', -1: '↓ niżej = bezpieczniej'}[DIRECTION[c]]
                    x[c] = st.slider(INFO[c]['label'], lo, hi, step=step, key=f'f_{c}',
                                     help=f'{INFO[c]["help"]}\n\n{good}. Mediana zdrowych firm: '
                                          f'{STATS[c]["healthy_median"]:.2f}, bankrutów: '
                                          f'{STATS[c]["bankrupt_median"]:.2f}.')

    scores = {h: float(s[0]) for h, s in score_frame(pd.DataFrame([x])).items()}
    pds = {h: float(calibrated_pd(scores[h], h)) for h in HORIZONS}
    pcts = {h: float(percentile(scores[h], h)) for h in HORIZONS}
    thr = {h: B['horizons'][h]['thresholds'][profile] for h in HORIZONS}
    alarms = {h: scores[h] >= thr[h] for h in HORIZONS}

    with right:
        h = main_h
        word, wcolor = risk_word(pcts[h])
        first_alarm = next((k for k in HORIZONS if alarms[k]), None)
        sub = f'{P["icon"]} {P["name"]} · horyzont {h_label(h)}'
        st.markdown(verdict_html(alarms[h], P['alarm'] if alarms[h] else P['ok'], sub), unsafe_allow_html=True)

        g1, g2 = st.columns([1, 1])
        with g1:
            fig = go.Figure(go.Indicator(
                mode='gauge+number', value=pcts[h],
                number={'suffix': ' / 100', 'font': {'size': 40}, 'valueformat': '.1f' if 99 <= pcts[h] < 100 else '.0f'},
                title={'text': f'Indeks ryzyka ({h_label(h)})<br><span style="font-size:12px;color:{GREY}">'
                               f'ryzykowniejsza niż {fmt_idx(pcts[h])}% firm</span>'},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': wcolor, 'thickness': .3},
                       'steps': [{'range': [0, 50], 'color': 'rgba(34,197,94,.25)'},
                                 {'range': [50, 80], 'color': 'rgba(234,179,8,.25)'},
                                 {'range': [80, 95], 'color': 'rgba(245,158,11,.35)'},
                                 {'range': [95, 100], 'color': 'rgba(239,68,68,.45)'}],
                       'threshold': {'line': {'color': 'white', 'width': 3}, 'thickness': .85,
                                     'value': float(percentile(thr[h], h))}}))
            plot(fig, height=260)
        with g2:
            st.markdown(f'#### Ryzyko: <span style="color:{wcolor}">{word}</span>', unsafe_allow_html=True)
            st.markdown(f'- Szacowane prawdopodobieństwo upadłości w horyzoncie **{h_label(h)}**: '
                        f'**{pds[h]:.2%}** (średnio dla rynku: {B["horizons"][h]["base_rate_test"]:.2%})')
            st.markdown(f'- To **{pds[h] / max(B["horizons"][h]["base_rate_test"], 1e-9):.1f}×** średnie ryzyko rynkowe')
            if first_alarm:
                st.markdown(f'- Pierwszy sygnał dla profilu {P["icon"]}: **horyzont {h_label(first_alarm)}**')
            else:
                st.markdown(f'- Brak sygnału w żadnym horyzoncie (1–{max(HORIZONS)} lat) dla profilu {P["icon"]}')
            st.caption('Biała kreska na wskaźniku = próg alarmu wybranego profilu.')

        # karty dla wszystkich horyzontów
        cols = st.columns(len(HORIZONS))
        for col, k in zip(cols, HORIZONS):
            colr = RED if alarms[k] else GREEN
            col.markdown(f'<div class="hcard"><div class="h">{h_label(k)}</div>'
                         f'<div class="v">{fmt_idx(pcts[k])}</div>'
                         f'<div class="s" style="color:{colr}">{"🚨 " + P["alarm_short"] if alarms[k] else "✅ OK"}</div>'
                         f'</div>', unsafe_allow_html=True)

        # krzywa ryzyka w czasie
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=HORIZONS, y=[pcts[k] for k in HORIZONS], mode='lines+markers',
                                 name='Indeks ryzyka firmy', line=dict(color=BLUE, width=4),
                                 marker=dict(size=14, color=[RED if alarms[k] else GREEN for k in HORIZONS],
                                             line=dict(color='white', width=2)),
                                 hovertemplate='%{x} lat: indeks %{y:.0f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=HORIZONS, y=[percentile(thr[k], k) for k in HORIZONS], mode='lines',
                                 name=f'Próg alarmu: {P["name"]}', line=dict(color='white', dash='dash')))
        fig.update_layout(title='Krzywa ryzyka w czasie', yaxis=dict(range=[0, 101], title='Indeks ryzyka (0–100)'),
                          xaxis=dict(title='Horyzont (lata)', tickvals=HORIZONS))
        plot(fig, height=300)

    st.divider()
    a, b = st.columns(2, gap='large')
    with a:
        st.markdown(f'### 🧠 Dlaczego taki wynik? ({h_label(main_h)})')
        contrib = contributions(x, main_h).sort_values()
        lbl = [INFO[c]['label'] for c in contrib.index]
        fig = go.Figure(go.Bar(x=contrib.values, y=lbl, orientation='h',
                               marker_color=[RED if v > 0 else GREEN for v in contrib.values],
                               hovertemplate='%{y}: %{x:+.2f}<extra></extra>'))
        fig.update_layout(xaxis_title='← obniża ryzyko   |   podnosi ryzyko →')
        plot(fig, height=380)
        st.caption('Wkład każdej cechy w wynik modelu (wartości SHAP liczone przez XGBoost).')
    with b:
        st.markdown('### 📍 Firma na tle rynku')
        fig = go.Figure()
        order = list(contrib.abs().sort_values().index)
        def pos(c, v):
            p = float(np.interp(v, STATS[c]['grid'], np.arange(101)))
            return p if DIRECTION[c] == 1 else 100 - p
        ylab = [INFO[c]['label'] for c in order]
        fig.add_trace(go.Scatter(x=[pos(c, STATS[c]['bankrupt_median']) for c in order], y=ylab, mode='markers',
                                 name='Typowy bankrut', marker=dict(symbol='x', size=12, color=RED)))
        fig.add_trace(go.Scatter(x=[pos(c, STATS[c]['healthy_median']) for c in order], y=ylab, mode='markers',
                                 name='Typowa zdrowa firma', marker=dict(symbol='diamond', size=12, color=GREEN)))
        fig.add_trace(go.Scatter(x=[pos(c, x[c]) for c in order], y=ylab, mode='markers', name='Ta firma',
                                 marker=dict(size=16, color=BLUE, line=dict(color='white', width=2))))
        fig.update_layout(xaxis=dict(range=[-2, 102], title='← gorzej niż rynek   |   lepiej niż rynek →'))
        plot(fig, height=380)
        st.caption('Pozycja każdego wskaźnika jako percentyl wszystkich firm, odwrócony tak, by „w prawo” było zawsze bezpieczniej.')

    st.markdown(f'### 🛠️ Co poprawić, żeby obniżyć ryzyko? ({h_label(main_h)})')
    base = scores[main_h]
    rows = []
    for c in FEATS:
        target = STATS[c]['healthy_median']
        if (target - x[c]) * DIRECTION[c] <= 0:
            continue
        x2 = dict(x)
        x2[c] = target
        s2 = float(score_frame(pd.DataFrame([x2]))[main_h][0])
        rows.append((c, target, s2))
    rows = sorted(rows, key=lambda r: r[2])[:3]
    if rows and rows[0][2] < base - 1e-4:
        cols = st.columns(len(rows))
        for col, (c, target, s2) in zip(cols, rows):
            with col.container(border=True):
                st.markdown(f'**{INFO[c]["label"]}**')
                st.markdown(f'{x[c]:.2f} → **{target:.2f}** <span class="pill">mediana zdrowych firm</span>',
                            unsafe_allow_html=True)
                st.metric('Indeks ryzyka', f'{percentile(s2, main_h):.0f}',
                          delta=f'{percentile(s2, main_h) - pcts[main_h]:+.0f}', delta_color='inverse')
                if alarms[main_h] and s2 < thr[main_h]:
                    st.success('Ta zmiana sama zdejmuje alarm')
    else:
        st.success('Wszystkie wskaźniki są już na poziomie zdrowych firm lub lepszym. 👍')


# ---------------------------------------------------------------- 2. portfel

with tab_port:
    st.markdown('### 📂 Ocena całego portfela')
    st.caption('Wgraj CSV z kolumnami cech (szablon poniżej) albo użyj przykładowego portfela 300 prawdziwych '
               'firm ze zbioru testowego (wzbogaconego o bankrutów, żeby było ich widać).')
    up_col, tpl_col = st.columns([3, 1])
    upload = up_col.file_uploader('Plik CSV', type=['csv'])
    tpl = pd.DataFrame([{**{'firma': 'Moja firma'}, **{c: round(STATS[c]['healthy_median'], 4) for c in FEATS}}])
    tpl_col.download_button('⬇️ Szablon CSV', tpl.to_csv(index=False).encode('utf-8'), 'szablon_portfel.csv',
                            'text/csv')

    port = None
    if upload is not None:
        try:
            port = pd.read_csv(upload)
            if not set(FEATS).issubset(port.columns):
                upload.seek(0)
                port = pd.read_csv(upload, sep=';')
            if not set(FEATS).issubset(port.columns):
                upload.seek(0)
                port = pd.read_csv(upload, sep=';', decimal=',')
        except Exception:
            pass

    if port is not None and not set(FEATS).issubset(port.columns):
        st.error(f'W pliku brakuje kolumn: {[c for c in FEATS if c not in port.columns]} — pokazuję portfel przykładowy.')
        port = None
    if port is not None:
        if 'firma' not in port.columns:
            port.insert(0, 'firma', [f'Firma {i + 1}' for i in range(len(port))])
        known = port['main_label'].values if 'main_label' in port.columns else None
    else:
        port = B['portfolio'].copy()
        known = port['main_label'].values

    S = score_frame(port)
    P_idx = {k: np.round(percentile(S[k], k), 1) for k in HORIZONS}
    view = pd.DataFrame({'Firma': port['firma']})
    view[f'Indeks ({h_label(main_h)})'] = P_idx[main_h]
    view['Status'] = np.where(S[main_h] >= B['horizons'][main_h]['thresholds'][profile],
                              f'🚨 {P["alarm"]}', '✅ OK')
    view['Krzywa ryzyka 1–6 lat'] = [list(v) for v in np.column_stack([P_idx[k] for k in HORIZONS])]
    view['Alarmy (na 6 horyzontów)'] = np.sum(
        [S[k] >= B['horizons'][k]['thresholds'][profile] for k in HORIZONS], axis=0)
    if known is not None:
        view['Faktycznie upadła (1 rok)'] = np.where(known == 1, '💀 tak', 'nie')
    view = view.sort_values(f'Indeks ({h_label(main_h)})', ascending=False)

    al = S[main_h] >= B['horizons'][main_h]['thresholds'][profile]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Firm w portfelu', f'{len(port)}')
    m2.metric(f'Alarmy ({h_label(main_h)})', f'{al.sum()}', help=f'Profil: {P["name"]}')
    m3.metric('Średni indeks ryzyka', f'{percentile(S[main_h], main_h).mean():.0f}')
    if known is not None and main_h == 1:
        m4.metric('Wyłapane bankructwa (1 rok)', f'{(al & (known == 1)).sum()} / {(known == 1).sum()}')
    elif known is not None:
        m4.metric('Bankructwa w portfelu (1 rok)', f'{(known == 1).sum()}')

    c1, c2 = st.columns([3, 2], gap='large')
    with c1:
        st.dataframe(view, hide_index=True, height=420, column_config={
            f'Indeks ({h_label(main_h)})': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%.1f'),
            'Krzywa ryzyka 1–6 lat': st.column_config.LineChartColumn(y_min=0, y_max=100),
        })
    with c2:
        fig = go.Figure()
        pp = percentile(S[main_h], main_h)
        fig.add_trace(go.Histogram(x=pp[~al], name='bez alarmu', marker_color=GREEN, xbins=dict(size=5)))
        fig.add_trace(go.Histogram(x=pp[al], name='alarm', marker_color=RED, xbins=dict(size=5)))
        fig.update_layout(barmode='stack', title=f'Rozkład indeksu ryzyka ({h_label(main_h)})',
                          xaxis_title='Indeks ryzyka', yaxis_title='Liczba firm')
        plot(fig, height=420)
    st.download_button('⬇️ Pobierz wyniki (CSV)', view.drop(columns='Krzywa ryzyka 1–6 lat').to_csv(index=False).encode('utf-8'), 'wyniki_portfel.csv',
                       'text/csv')


# ---------------------------------------------------------------- 3. wartość biznesowa

with tab_value:
    st.markdown(f'### 💼 Ile to jest warte? — {P["icon"]} {P["name"]}, horyzont {h_label(main_h)}')
    H = B['horizons'][main_h]
    m = H['profiles'][profile]
    br = H['base_rate_test']
    st.caption(f'Liczby z prawdziwej ewaluacji na zbiorze testowym ({H["split"]["n_test"]:,} firm-lat, '
               f'lata {H["split"]["test"]}). Zmień profil w panelu bocznym, żeby porównać.')

    if profile in ('bank', 'balanced'):
        i1, i2, i3, i4 = st.columns(4)
        n_app = i1.number_input('Wniosków kredytowych / rok', 100, 1_000_000, 10_000, step=1000)
        loan = i2.number_input('Średnia kwota kredytu (zł)', 10_000, 100_000_000, 500_000, step=50_000)
        lgd = i3.slider('Strata przy bankructwie (LGD)', 0.1, 1.0, 0.45, 0.05)
        cost = i4.number_input('Koszt ręcznej analizy wniosku (zł)', 0, 100_000, 800, step=100)
        n_bad = n_app * br
        caught = n_bad * m['recall']
        reviews = n_app * m['alert_rate']
        saved = caught * loan * lgd
        spent = reviews * cost
        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Przyszłych bankrutów wśród wniosków', f'{n_bad:,.0f}')
        k2.metric('Wyłapanych przez model', f'{caught:,.0f}', f'{m["recall"]:.0%}')
        k3.metric('Wniosków zatwierdzanych automatycznie', f'{n_app - reviews:,.0f}', f'{1 - m["alert_rate"]:.0%}')
        k4.metric('Wniosków do analityka', f'{reviews:,.0f}', f'{m["alert_rate"]:.0%}', delta_color='off')
        fig = go.Figure(go.Waterfall(
            x=['Uniknięte straty kredytowe', 'Koszt weryfikacji alarmów', 'Wynik netto'],
            measure=['relative', 'relative', 'total'], y=[saved, -spent, 0],
            text=[f'{saved / 1e6:,.1f} mln zł', f'-{spent / 1e6:,.1f} mln zł', f'{(saved - spent) / 1e6:,.1f} mln zł'],
            textposition='outside', increasing=dict(marker_color=GREEN), decreasing=dict(marker_color=RED),
            totals=dict(marker_color=BLUE)))
        fig.update_layout(title='Roczny efekt finansowy (szacunek)', yaxis_title='zł')
        plot(fig, height=380)
        st.info(f'Bank zatwierdza automatycznie **{1 - m["alert_rate"]:.0%}** wniosków, a do analityka trafia '
                f'**{m["alert_rate"]:.0%}** — wśród nich {m["recall"]:.0%} wszystkich przyszłych bankrutów.')
    else:
        i1, i2, i3, i4 = st.columns(4)
        universe = i1.number_input('Liczba obserwowanych spółek', 100, 1_000_000, 5_000, step=500)
        pos_size = i2.number_input('Kapitał na pozycję (zł)', 1_000, 100_000_000, 1_000_000, step=100_000)
        gain = i3.slider('Zysk na trafionym shorcie', 0.1, 1.0, 0.6, 0.05, help='Spadek kursu spółki, która upada.')
        loss = i4.slider('Strata na nietrafionym shorcie', 0.0, 0.5, 0.15, 0.01,
                         help='Średni ruch przeciw pozycji + koszty pożyczki akcji.')
        signals = universe * m['alert_rate']
        hits = signals * m['precision']
        ev = m['precision'] * gain - (1 - m['precision']) * loss
        breakeven = loss / (gain + loss)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Sygnałów short / rok', f'{signals:,.0f}')
        k2.metric('Trafione (spółka upada)', f'{hits:,.0f}', f'trafność {m["precision"]:.1%}')
        k3.metric('Próg opłacalności (trafność)', f'{breakeven:.1%}')
        k4.metric('Oczekiwany zwrot na pozycję', f'{ev:+.1%}', delta_color='normal')
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Trafność modelu', 'Wymagana do zysku'], y=[m['precision'], breakeven],
                             marker_color=[GREEN if m['precision'] >= breakeven else RED, GREY],
                             text=[f'{m["precision"]:.1%}', f'{breakeven:.1%}'], textposition='outside'))
        fig.update_layout(title='Czy strategia zarabia?', yaxis=dict(tickformat='.0%'))
        plot(fig, height=340)
        verdict = 'zarabia' if ev > 0 else 'nie zarabia'
        st.info(f'Przy tych założeniach strategia **{verdict}**: oczekiwany wynik portfela ≈ '
                f'**{ev * pos_size * signals / 1e6:,.1f} mln zł** rocznie. Dla porównania losowy wybór spółek '
                f'miałby trafność {br:.2%}, czyli model jest **{m["precision"] / max(br, 1e-9):.0f}×** lepszy.')


# ---------------------------------------------------------------- 4. jakość modelu

with tab_quality:
    st.markdown('### 🧪 Jak dobry jest model? (zbiór testowy, którego model nigdy nie widział)')
    rows = []
    for k in HORIZONS:
        Hk = B['horizons'][k]
        for p_name, p in PROFILES.items():
            mm = Hk['profiles'][p_name]
            rows.append(dict(h=k, profil=f'{p["icon"]} {p["name"]}', recall=mm['recall'], precision=mm['precision'],
                             color=p['color']))
    df_q = pd.DataFrame(rows)

    c1, c2 = st.columns(2, gap='large')
    with c1:
        fig = go.Figure()
        for p_name, p in PROFILES.items():
            d = df_q[df_q['profil'] == f'{p["icon"]} {p["name"]}']
            fig.add_trace(go.Scatter(x=d['h'], y=d['recall'], name=f'{p["icon"]} {p["name"]}', mode='lines+markers',
                                     line=dict(color=p['color'], width=3), marker=dict(size=9)))
        fig.update_layout(title='Wykrywalność (recall) — jaki % bankrutów łapiemy',
                          yaxis=dict(tickformat='.0%', range=[0, 1.05]), xaxis=dict(title='Horyzont (lata)', tickvals=HORIZONS))
        plot(fig, height=340)
    with c2:
        fig = go.Figure()
        for p_name, p in PROFILES.items():
            d = df_q[df_q['profil'] == f'{p["icon"]} {p["name"]}']
            fig.add_trace(go.Scatter(x=d['h'], y=d['precision'], name=f'{p["icon"]} {p["name"]}', mode='lines+markers',
                                     line=dict(color=p['color'], width=3), marker=dict(size=9)))
        fig.add_trace(go.Scatter(x=HORIZONS, y=[B['horizons'][k]['base_rate_test'] for k in HORIZONS],
                                 name='Losowy wybór', line=dict(color=GREY, dash='dot')))
        fig.update_layout(title='Trafność (precision) — jaki % alarmów się sprawdza',
                          yaxis=dict(tickformat='.0%'), xaxis=dict(title='Horyzont (lata)', tickvals=HORIZONS))
        plot(fig, height=340)

    auc_df = pd.DataFrame({
        'Horyzont': [h_label(k) for k in HORIZONS],
        'ROC-AUC': [B['horizons'][k]['roc_auc'] for k in HORIZONS],
        'PR-AUC': [B['horizons'][k]['pr_auc'] for k in HORIZONS],
        'Odsetek bankrutów': [B['horizons'][k]['base_rate_test'] for k in HORIZONS],
        'Firm-lat w teście': [B['horizons'][k]['split']['n_test'] for k in HORIZONS],
        'Bankrutów w teście': [B['horizons'][k]['split']['pos_test'] for k in HORIZONS],
    })
    auc_df['Przewaga nad losowaniem (PR-AUC / odsetek)'] = auc_df['PR-AUC'] / auc_df['Odsetek bankrutów']
    st.dataframe(auc_df.style.format({'ROC-AUC': '{:.3f}', 'PR-AUC': '{:.3f}', 'Odsetek bankrutów': '{:.2%}',
                                      'Firm-lat w teście': '{:,}', 'Bankrutów w teście': '{:,}',
                                      'Przewaga nad losowaniem (PR-AUC / odsetek)': '{:.0f}×'}),
                 hide_index=True)

    st.markdown(f'#### 🎚️ Zagraj progiem — horyzont {h_label(main_h)}')
    H = B['horizons'][main_h]
    pr = pd.DataFrame(H['pr_curve'])
    t_pos = st.slider('Próg alarmu (indeks ryzyka)', 0.0, 99.9, min(float(percentile(H['thresholds'][profile], main_h)), 99.9), step=0.1,
                      help='Przesuń, żeby zobaczyć kompromis: niższy próg = więcej wykrytych bankrutów, '
                           'ale więcej fałszywych alarmów.')
    t_val = float(np.interp(t_pos, np.arange(1001) / 10, H['q']))
    row = pr.iloc[(pr['threshold'] - t_val).abs().argmin()]
    n = 10_000
    n_bad = n * H['base_rate_test']
    tp = n_bad * row['recall']
    fp = tp / max(row['precision'], 1e-9) - tp
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Wykrywalność', f'{row["recall"]:.1%}')
    k2.metric('Trafność', f'{row["precision"]:.1%}')
    k3.metric('Na 10 000 firm: wykryci bankruci', f'{tp:,.0f} z {n_bad:,.0f}')
    k4.metric('Na 10 000 firm: fałszywe alarmy', f'{fp:,.0f}')
    for p_name, p in PROFILES.items():
        st.caption(f'{p["icon"]} {p["name"]}: próg = indeks {fmt_idx(percentile(H["thresholds"][p_name], main_h))}')

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = go.Figure(go.Scatter(x=H['roc_curve']['fpr'], y=H['roc_curve']['tpr'], line=dict(color=BLUE, width=3),
                                   name='Model'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=GREY, dash='dot'), name='Losowy'))
        fig.update_layout(title=f'Krzywa ROC (AUC = {H["roc_auc"]:.3f})', xaxis_title='Fałszywe alarmy (FPR)',
                          yaxis_title='Wykrywalność (TPR)', showlegend=False)
        plot(fig, height=320)
    with c2:
        fig = go.Figure(go.Scatter(x=pr['recall'], y=pr['precision'], line=dict(color=VIOLET, width=3)))
        fig.add_trace(go.Scatter(x=[row['recall']], y=[row['precision']], mode='markers',
                                 marker=dict(size=14, color='white')))
        fig.update_layout(title=f'Precision–Recall (AP = {H["pr_auc"]:.3f})', xaxis_title='Wykrywalność',
                          yaxis_title='Trafność', showlegend=False)
        plot(fig, height=320)
    with c3:
        bins = np.asarray(H['hist']['bins'])
        mid = (bins[1:] + bins[:-1]) / 2
        fig = go.Figure()
        fig.add_trace(go.Bar(x=mid, y=H['hist']['healthy'], name='Zdrowe', marker_color=GREEN, opacity=.7))
        fig.add_trace(go.Bar(x=mid, y=H['hist']['bankrupt'], name='Bankruci', marker_color=RED, opacity=.7))
        fig.add_vline(x=t_val, line=dict(color='white', dash='dash'))
        fig.update_layout(barmode='overlay', title='Rozkład wyniku modelu', xaxis_title='Wynik modelu',
                          yaxis_title='Gęstość')
        plot(fig, height=320)


# ---------------------------------------------------------------- 5. o projekcie

with tab_about:
    
    st.markdown("""
### 🎯 Cel biznesowy projektu

**Problem:** Tradycyjne metody oceny ryzyka finansowego reagują z opóźnieniem. Analiza setek wskaźników jest czasochłonna dla analityków, a wczesne sygnały upadłości giną w "szumie" informacyjnym.
<br>
**Rozwiązanie:** Zautomatyzowany Radar Upadłości bazujący na uczeniu maszynowym. Analizuje on surowe sprawozdania finansowe i z wyprzedzeniem (od 1 do 6 lat) wyłapuje ukryte wzorce prowadzące do bankructwa. 
<br>
**Zastosowanie:** 
- 🏦 **Banki** (minimalizacja portfela niespłacanych kredytów, czyli NPL - Non-Performing Loans)
- 📉 **Fundusze inwestycyjne** (wyszukiwanie spółek do shortowania na giełdzie)
- 🤝 **B2B** (zabezpieczenie łańcuchów dostaw przed upadkiem kluczowego kontrahenta)

---

### 🧠 Architektura i wyzwania modelowania

**1. Skrajna nierównowaga klas (Data Imbalance)**
W zbiorze V4FinBench (obejmującym milion raportów z Europy Środkowej) bankructwa stanowią zaledwie ok. **0,3%** obserwacji. 
* **Downsampling:** Zastosowaliśmy ręczny podział w proporcji 1 bankrut : 5 zdrowych firm. Wymusza to na modelu skupienie się na rzadkiej klasie mniejszościowej.
* **Kalibracja prawdopodobieństw:** Po wytrenowaniu modelu zastosowaliśmy dodatkową kalibrację. Dzięki temu surowy scoring generowany przez XGBoost został zamieniony w rzeczywiste, godne zaufania prawdopodobieństwa (tzw. wykres *Reliability Diagram* przedstawiony poniżej).
""")
    # Wrap in columns to make the image smaller and centered
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(str(Path(__file__).parent / 'calibration.png'), caption='Wykres kalibracji (Reliability Diagram)')
    st.markdown("""

**2. Zapobieganie wyciekom danych (Data Leakage)**
Zrezygnowaliśmy z klasycznego losowego podziału (Train/Test Split). Zastosowaliśmy rygorystyczny podział **czasowy**:
- Model uczy się wyłącznie na latach przeszłych (np. do 2015 r.).
- Progi odcięcia dobierane są na roku 2016 (Walidacja).
- Wyniki na demonstracji pochodzą wyłącznie z lat późniejszych, których model *nigdy* wcześniej nie widział. 
Symuluje to rzeczywiste warunki produkcyjne.

**3. Hiperparametryzacja**
Wykorzystaliśmy RandomizedSearchCV do przeszukania przestrzeni hiperparametrów XGBoosta. Pozwoliło nam to zbalansować głębokość drzew (max_depth) z tempem uczenia (learning_rate) oraz zabezpieczyć model przed przeuczeniem za pomocą odpowiedniego min_child_weight.

---

""")
    st.markdown("""
#### 💡 Dlaczego wybraliśmy dokładnie 10 cech?

Nasze analizy w notatnikach wykazały, że początkowy zestaw ponad 130 wskaźników finansowych jest wysoce skorelowany i nadmiarowy. Przeprowadziliśmy eksperyment redukcji wymiarów (XGBoost Feature Selection), z którego jednoznacznie wynikło, że ograniczenie modelu do 10-15 najważniejszych cech powoduje spadek metryki ROC-AUC o marginalne wartości rzędu **~0.007** (np. z 0.908 do 0.901). Zdecydowaliśmy się na 10 cech, co drastycznie uprościło wdrażanie modelu i zminimalizowało ryzyko przeuczenia, zachowując niemal 99% pierwotnej siły predykcyjnej.
""")

    st.dataframe(pd.DataFrame([
        {"Liczba cech": "Wszystkie (133)", "ROC-AUC na teście": "0.9086"},
        {"Liczba cech": "50", "ROC-AUC na teście": "0.9063"},
        {"Liczba cech": "30", "ROC-AUC na teście": "0.9043"},
        {"Liczba cech": "20", "ROC-AUC na teście": "0.9019"},
        {"Liczba cech": "15", "ROC-AUC na teście": "0.9010"},
        {"Liczba cech": "10", "ROC-AUC na teście": "0.9046 (Wybrany kompromis)"}
    ]), use_container_width=True, hide_index=True)

    st.markdown("""
#### 💡 Dlaczego zdecydowaliśmy się na model XGBoost?

Sprawozdania finansowe (jak zbiór V4FinBench) są bardzo "brudne". Zawierają tysiące braków danych, nieciągłości (NaN) i potężne wartości odstające (outliery). XGBoost jako algorytm oparty na drzewach decyzyjnych genialnie i automatycznie radzi sobie z brakami (uczy się, w którą gałąź skierować brakującą wartość) oraz całkowicie ignoruje wartości odstające, które rozbiłyby tradycyjne modele regresji logistycznej. Dodatkowo wbudowana parametryzacja ułatwiła nam naukę na wybitnie niezbalansowanych klasach (bankruci to ułamek procenta populacji).
""")

    st.dataframe(pd.DataFrame([
        {"Model": "Random Forest", "ROC-AUC": "0.9146", "F1-Score": "0.1571", "Recall": "0.2143", "Precision": "0.1240"},
        {"Model": "XGBoost (Baseline)", "ROC-AUC": "0.9079", "F1-Score": "0.1263", "Recall": "0.1848", "Precision": "0.0959"},
        {"Model": "Logistic Regression", "ROC-AUC": "0.8181", "F1-Score": "0.0555", "Recall": "0.1996", "Precision": "0.0323"}
    ]), use_container_width=True, hide_index=True)

    st.markdown("""
Widzimy, że Random Forest oraz XGBoost deklasują Regresję Logistyczną. Ostatecznie wybraliśmy wariant XGBoost ze względu na znacznie wyższą szybkość predykcji, natywne wsparcie dla braków danych (bez konieczności dodatkowej imputacji, której wymagał Random Forest) oraz lepszą kontrolę nad downsamplingiem przy skrajnie rzadkiej klasie bankrutów.
""")
    st.markdown('#### Cechy używane przez model')
    st.dataframe(pd.DataFrame([{
        'Kategoria': f'{CATEGORY_ICONS[INFO[c]["category"]]} {INFO[c]["category"]}',
        'Cecha': INFO[c]['label'],
        'Co oznacza': INFO[c]['help'],
        'Zdrowe (mediana)': round(STATS[c]['healthy_median'], 3),
        'Bankruci (mediana)': round(STATS[c]['bankrupt_median'], 3),
        'Kolumna w danych': c,
    } for c in FEATS]), hide_index=True)
    st.markdown('#### Podział danych i parametry modeli')
    st.dataframe(pd.DataFrame([{
        'Horyzont': h_label(k), **{f'{n}': v for n, v in B['horizons'][k]['split'].items()},
        'Parametry': ', '.join(f'{a}={b}' for a, b in B['horizons'][k]['best_params'].items()),
    } for k in HORIZONS]), hide_index=True)
