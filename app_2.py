# ============================================================
# app_2.py — aplikacja Streamlit dla bundle hybrydowego
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ---------------- KONFIGURACJA ----------------
st.set_page_config(
    page_title="Predykcja Bankructwa",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

KATALOG_DANYCH      = "data"
KATALOG_PRZETWORZONY = os.path.join(KATALOG_DANYCH, "processed")
SCIEZKA_BUNDLE      = os.path.join(KATALOG_PRZETWORZONY, "models_bundle_hybrid.pkl")

OPIS_HORYZONTU = {
    "h1": "1 rok przed bankructwem",
    "h2": "2 lata przed bankructwem",
    "h3": "3 lata przed bankructwem",
    "h4": "4 lata przed bankructwem",
    "h5": "5 lat przed bankructwem",
    "h6": "6 lat przed bankructwem",
}

ETYKIETA_WIARYGODNOSCI = {
    "high":   "🟢 wysoka",
    "medium": "🟡 średnia",
    "low":    "🔴 niska",
}


# ============================================================
# WCZYTANIE BUNDLE
# ============================================================
@st.cache_resource
def wczytaj_bundle():
    """Wczytuje bundle z modelami (cache'owane na poziomie zasobów)."""
    if not os.path.exists(SCIEZKA_BUNDLE):
        st.error(f"❌ Nie znaleziono pliku bundle: {SCIEZKA_BUNDLE}")
        return None
    return joblib.load(SCIEZKA_BUNDLE)


# ============================================================
# WCZYTANIE DANYCH DO STATYSTYK / WARTOŚCI DOMYŚLNYCH
# ============================================================
@st.cache_data
def wczytaj_statystyki(h):
    """Wczytuje dane dla danego horyzontu (do zakładki statystyk)."""
    sciezka = os.path.join(KATALOG_DANYCH, f"company_years_{h}.parquet")
    df = pd.read_parquet(sciezka)
    df = df.dropna(subset=["main_label"]).copy()
    df["main_label"] = df["main_label"].astype(int)
    return df


@st.cache_data
def wczytaj_wartosci_domyslne(h):
    """Mediany dla UI — tylko dla cech modelu (nie dla wszystkich kolumn)."""
    bundle = wczytaj_bundle()
    art = bundle["models"].get(h)
    if art is None:
        return {}
    mediany = art["median_fill"]
    return {f: float(mediany.get(f, 0.0)) for f in art["features"]}


# ============================================================
# PREDYKCJA
# ============================================================
def przewiduj(art, wartosci_cech: dict):
    """Zwraca (prawdopodobieństwo, predykcja 0/1)."""
    cechy = art["features"]
    mediany = art["median_fill"]

    # Budujemy wektor w kolejności cech; braki uzupełniamy medianami
    wiersz = {}
    for f in cechy:
        v = wartosci_cech.get(f, np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = mediany.get(f, 0.0)
        wiersz[f] = v

    X = pd.DataFrame([wiersz], columns=cechy)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # XGBoost zwraca prawdopodobieństwa bezpośrednio (nie margin)
    proba = float(art["model"].predict_proba(X)[0, 1])
    pred = int(proba >= art["threshold"])
    return proba, pred


# ============================================================
# INTERFEJS
# ============================================================
st.title("🏦 Predykcja Bankructwa — Hybryda")
st.markdown(
    "**Modele:** XGBoost · **Dane:** V4FinBench · "
    "**Horyzonty:** h1–h6 · **Bundle:** hybrid-1.0"
)

bundle = wczytaj_bundle()
if bundle is None:
    st.stop()

dostepne_horyzonty = [h for h in bundle["horizons"] if h in bundle["models"]]
if not dostepne_horyzonty:
    st.error("❌ Brak dostępnych modeli w bundle")
    st.stop()

# ============================================================
# PANEL BOCZNY
# ============================================================
st.sidebar.header("⚙️ Konfiguracja")

wybrany_horyzont = st.sidebar.selectbox(
    "📅 Horyzont",
    dostepne_horyzonty,
    index=0,
    format_func=lambda h: f"{h.upper()} — {OPIS_HORYZONTU[h]}",
)

art = bundle["models"][wybrany_horyzont]
m = art["metrics"]

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Metryki modelu")
st.sidebar.metric("PR-AUC", f"{m['pr_auc']:.4f}",
                  help=f"lift {m['pr_auc_lift']:.1f}× baseline")
st.sidebar.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
st.sidebar.metric("Recall (Czułość)", f"{m['recall']:.4f}")
st.sidebar.metric("Precision (Precyzja)", f"{m['precision']:.4f}")
st.sidebar.metric("F2 (β=2)", f"{m['fbeta']:.4f}")

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Parametry")
st.sidebar.write(f"**Źródło:** `{art['source']}` / `{art['source_key']}`")
st.sidebar.write(f"**Liczba cech (top-N):** {art['top_n']}")
st.sidebar.write(f"**Bez flagi insolvency:** {art['drop_insolvency']}")
st.sidebar.write(f"**Użyto lagów:** {art['use_lags']}")
st.sidebar.write(f"**Próg decyzyjny:** {art['threshold']:.4f}")
st.sidebar.write(f"**PR-AUC lift:** {m['pr_auc_lift']:.1f}×")
st.sidebar.write(f"**Wiarygodność:** {ETYKIETA_WIARYGODNOSCI.get(art['reliability'], '—')}")

if art["reliability"] == "low":
    st.sidebar.warning(
        "⚠️ Model ma **niski** PR-AUC lift. "
        "Należy traktować wyłącznie jako sygnał pomocniczy."
    )

# ============================================================
# ZAKŁADKI
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔮 Predykcja", "📊 Statystyki", "📖 O modelach"])

# ---------- ZAKŁADKA 1: PREDYKCJA ----------
with tab1:
    st.header(f"🔮 Predykcja — {wybrany_horyzont.upper()}")
    st.caption(
        f"Wprowadź wartości **{len(art['features'])} cech**. "
        f"Próg modelu: **{art['threshold']:.4f}**. "
        f"Braki zostaną uzupełnione medianami ze zbioru treningowego."
    )

    kol_btn1, kol_btn2 = st.columns([1, 3])
    with kol_btn1:
        if st.button("📋 Wypełnij przykładem", use_container_width=True):
            df_probka = wczytaj_statystyki(wybrany_horyzont)
            wiersz = df_probka.sample(1).iloc[0]
            for f in art["features"]:
                if f in wiersz.index:
                    st.session_state[f"in_{f}"] = float(wiersz[f])
            st.rerun()

    domyslne = wczytaj_wartosci_domyslne(wybrany_horyzont)

    with st.form("formularz_predykcji"):
        st.subheader("📝 Wprowadź cechy firmy")
        n_kol = 3
        kolumny = st.columns(n_kol)
        wartosci_wejsciowe = {}

        for i, f in enumerate(art["features"]):
            with kolumny[i % n_kol]:
                etykieta = f if len(f) <= 40 else f[:37] + "..."
                wartosci_wejsciowe[f] = st.number_input(
                    etykieta,
                    value=float(domyslne.get(f, 0.0)),
                    format="%.6f",
                    key=f"in_{f}",
                )

        zatwierdzono = st.form_submit_button(
            "🚀 Przewiduj", use_container_width=True
        )

    if zatwierdzono:
        proba, pred = przewiduj(art, wartosci_wejsciowe)

        st.markdown("---")
        st.subheader("📊 Wynik predykcji")
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Prawdopodobieństwo", f"{proba*100:.2f}%")
        with k2:
            st.metric("Klasyfikacja",
                      "🔴 BANKRUCTWO" if pred else "🟢 BRAK BANKRUCTWA")
        with k3:
            st.metric("Próg", f"{art['threshold']:.4f}")

        st.progress(min(max(proba, 0.0), 1.0))

        if pred:
            st.error(
                f"⚠️ **Wysokie ryzyko bankructwa**\n\n"
                f"- Prawdopodobieństwo: **{proba*100:.2f}%**\n"
                f"- Próg decyzyjny: **{art['threshold']:.4f}**\n"
                f"- Horyzont: **{OPIS_HORYZONTU[wybrany_horyzont]}**\n"
                f"- Wiarygodność modelu: "
                f"**{ETYKIETA_WIARYGODNOSCI.get(art['reliability'])}**"
            )
        else:
            st.success(
                f"✅ **Niskie ryzyko bankructwa**\n\n"
                f"- Prawdopodobieństwo: **{proba*100:.2f}%**\n"
                f"- Próg decyzyjny: **{art['threshold']:.4f}**"
            )

# ---------- ZAKŁADKA 2: STATYSTYKI ----------
with tab2:
    st.header(f"📊 Statystyki — {wybrany_horyzont.upper()}")

    with st.spinner("Ładowanie danych..."):
        df_stat = wczytaj_statystyki(wybrany_horyzont)

    n_total = len(df_stat)
    n_bank = int((df_stat["main_label"] == 1).sum())
    n_zdrowe = int((df_stat["main_label"] == 0).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Wszystkich firm", f"{n_total:,}")
    k2.metric("Bankructwa", f"{n_bank:,}")
    k3.metric("Zdrowe firmy", f"{n_zdrowe:,}")
    k4.metric("Proporcja klas", f"{n_zdrowe/max(n_bank,1):.0f}:1")

    st.markdown("---")
    st.subheader("📈 Rozkład klas")
    fig, ax = plt.subplots(figsize=(8, 4))
    liczby = df_stat["main_label"].value_counts().sort_index()
    slupki = ax.bar(["Zdrowe", "Bankructwa"], liczby.values,
                    color=["green", "red"], alpha=0.7)
    for s, v in zip(slupki, liczby.values):
        ax.text(s.get_x() + s.get_width()/2, s.get_height(),
                f"{v:,}", ha="center", va="bottom")
    ax.set_ylabel("Liczba firm")
    ax.set_title(f"Rozkład klas — {wybrany_horyzont.upper()}")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📊 Porównanie horyzontów")
    wiersze = []
    for h in dostepne_horyzonty:
        mm = bundle["models"][h]["metrics"]
        wiersze.append({
            "Horyzont": h.upper(),
            "PR-AUC": mm["pr_auc"],
            "PR-AUC lift": mm["pr_auc_lift"],
            "ROC-AUC": mm["roc_auc"],
            "Recall": mm["recall"],
            "Precision": mm["precision"],
            "F2": mm["fbeta"],
            "Wiarygodność": ETYKIETA_WIARYGODNOSCI.get(
                bundle["models"][h]["reliability"], "—"),
        })
    df_por = pd.DataFrame(wiersze)

    fig, osie = plt.subplots(1, 3, figsize=(15, 4))
    osie[0].bar(df_por["Horyzont"], df_por["PR-AUC lift"], color="steelblue")
    osie[0].set_title("PR-AUC lift (× baseline)")
    osie[0].axhline(10, color="orange", linestyle="--", linewidth=1)
    osie[0].axhline(30, color="green", linestyle="--", linewidth=1)
    osie[0].grid(True, alpha=0.3)

    osie[1].bar(df_por["Horyzont"], df_por["ROC-AUC"], color="green")
    osie[1].set_title("ROC-AUC")
    osie[1].grid(True, alpha=0.3)

    osie[2].bar(df_por["Horyzont"], df_por["Recall"], color="orange")
    osie[2].set_title("Recall (Czułość)")
    osie[2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(df_por, use_container_width=True, hide_index=True)

# ---------- ZAKŁADKA 3: O MODELACH ----------
with tab3:
    st.header("📖 O modelach")
    st.markdown(
        "**Bundle:** hybrid-1.0 — kombinacja modeli z v2.0 i v3.0.\n\n"
        "**Zasada:** dla każdego horyzontu wybrano najlepszy model "
        "według PR-AUC lift."
    )

    wiersze_wyboru = []
    for h in dostepne_horyzonty:
        a = bundle["models"][h]
        wiersze_wyboru.append({
            "Horyzont": h.upper(),
            "Źródło": a["source"],
            "Klucz": a["source_key"],
            "Top-N": a["top_n"],
            "Lagi": a["use_lags"],
            "Bez flagi insolvency": a["drop_insolvency"],
            "PR-AUC lift": a["metrics"]["pr_auc_lift"],
            "Wiarygodność": ETYKIETA_WIARYGODNOSCI.get(a["reliability"], "—"),
        })
    st.dataframe(pd.DataFrame(wiersze_wyboru),
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(f"🏆 Top-20 cech — {wybrany_horyzont.upper()}")
    top20 = art["features"][:20]
    st.dataframe(
        pd.DataFrame({"#": range(1, len(top20)+1), "Cecha": top20}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    st.subheader("📋 Wszystkie metryki wybranego modelu")
    tabela_metryk = pd.DataFrame({
        "Metryka": [
            "PR-AUC", "PR-AUC baseline", "PR-AUC lift",
            "ROC-AUC", "Recall (Czułość)", "Precision (Precyzja)",
            "F1", "F2 (β=2)", "Specificity (Swoistość)", "NPV",
            "TP", "FP", "FN", "TN",
            "Bankructwa w zbiorze testowym",
            "Bankructwa bez flagi Insolvency_flag",
        ],
        "Wartość": [
            m["pr_auc"], m["baseline_pr_auc"], m["pr_auc_lift"],
            m["roc_auc"], m["recall"], m["precision"],
            m["f1"], m["fbeta"], m["specificity"], m["npv"],
            m["TP"], m["FP"], m["FN"], m["TN"],
            m["n_bank_test"], m["n_bank_test_with_flag0"],
        ],
    })
    st.dataframe(tabela_metryk, use_container_width=True, hide_index=True)

# ---------- STOPKA ----------
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**🏦 Predykcja Bankructwa — Hybryda**\n\n"
    "Bundle: `models_bundle_hybrid.pkl`  \n"
    "Powered by Streamlit"
)

# uruchom streamlit:
# C:\Users\????\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run app.py