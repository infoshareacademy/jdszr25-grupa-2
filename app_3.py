# -*- coding: utf-8 -*-
# ============================================================
# app_2.py — aplikacja Streamlit dla bundle hybrydowego
# Wersja 2.0 — z kalibracją i wyborem bundle
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# ---------------- KONFIGURACJA ----------------
st.set_page_config(
    page_title="Predykcja Bankructwa",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ✅ PEŁNE ŚCIEŻKI (naprawiony błąd)
KATALOG_DANYCH       = "data"
KATALOG_PRZETWORZONY = os.path.join(KATALOG_DANYCH, "processed")

# ✅ DWA BUNDLE — oryginalny i skalibrowany
BUNDLE_ORYGINALNY   = os.path.join(KATALOG_PRZETWORZONY, "models_bundle_hybrid.pkl")
BUNDLE_SKALIBROWANY = os.path.join(KATALOG_PRZETWORZONY, "models_bundle_hybrid_calibrated.pkl")

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
def wczytaj_bundle(sciezka):
    """Wczytuje bundle z modelami (cache'owane na poziomie zasobów)."""
    if not os.path.exists(sciezka):
        st.error(f"❌ Nie znaleziono pliku bundle: {sciezka}")
        return None
    return joblib.load(sciezka)


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
def wczytaj_wartosci_domyslne(sciezka_bundle, h):
    """Mediany dla UI — tylko dla cech modelu (nie dla wszystkich kolumn)."""
    bundle = joblib.load(sciezka_bundle)
    art = bundle["models"].get(h)
    if art is None:
        return {}
    mediany = art["median_fill"]
    return {f: float(mediany.get(f, 0.0)) for f in art["features"]}


# ============================================================
# PREDYKCJA Z KALIBRACJĄ
# ============================================================
def przewiduj(art, wartosci_cech: dict):
    """
    Zwraca (prawdopodobieństwo, predykcja 0/1, is_calibrated).
    
    Jeśli model ma kalibrator — używa go do uzyskania prawdziwych prawdopodobieństw.
    W przeciwnym razie zwraca surowy score.
    """
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

    # Surowy score z modelu
    proba_raw = float(art["model"].predict_proba(X)[0, 1])
    
    # ✅ KALIBRACJA (jeśli dostępna)
    if art.get("calibrator") is not None:
        proba = float(art["calibrator"].predict([proba_raw])[0])
        is_calibrated = True
    else:
        proba = proba_raw
        is_calibrated = False
    
    pred = int(proba >= art["threshold"])
    return proba, pred, is_calibrated


# ============================================================
# INTERFEJS
# ============================================================
st.title("🏦 Predykcja Bankructwa — Hybryda")
st.markdown(
    "**Modele:** XGBoost · **Dane:** V4FinBench · "
    "**Horyzonty:** h1–h6"
)

# ============================================================
# SIDEBAR — WYBÓR BUNDLE
# ============================================================
st.sidebar.header("⚙️ Konfiguracja")

# ✅ WYBÓR BUNDLE
# ✅ WYBÓR BUNDLE — SKALIBROWANE DOMYŚLNIE
st.sidebar.subheader("📦 Wersja modeli")
bundle_wybor = st.sidebar.radio(
    "Wybierz bundle",
    ["Skalibrowane (prawdziwe prob.)", "Oryginalne (lepszy ranking)"],  # ← zamieniona kolejność
    index=0,  # ← 0 = pierwszy element (Skalibrowane)
    help=(
        "Skalibrowane — prawdziwe prawdopodobieństwa (Brier score), "
        "lekko gorszy ranking\n"
        "Oryginalne — lepszy ROC-AUC/PR-AUC, ale score nie jest prawdopodobieństwem"
    ),
)

if bundle_wybor == "Skalibrowane (prawdziwe prob.)":
    SCIEZKA_BUNDLE = BUNDLE_SKALIBROWANY
    is_calibrated_bundle = True
else:
    SCIEZKA_BUNDLE = BUNDLE_ORYGINALNY
    is_calibrated_bundle = False

# Sprawdź czy plik istnieje
if not os.path.exists(SCIEZKA_BUNDLE):
    st.error(f"❌ Brak pliku: {SCIEZKA_BUNDLE}")
    st.info("Uruchom najpierw odpowiedni skrypt treningowy/kalibracyjny.")
    st.stop()

bundle = wczytaj_bundle(SCIEZKA_BUNDLE)
if bundle is None:
    st.stop()

dostepne_horyzonty = [h for h in bundle["horizons"] if h in bundle["models"]]
if not dostepne_horyzonty:
    st.error("❌ Brak dostępnych modeli w bundle")
    st.stop()

# ============================================================
# WYBÓR HORYZONTU
# ============================================================
wybrany_horyzont = st.sidebar.selectbox(
    "📅 Horyzont",
    dostepne_horyzonty,
    index=0,
    format_func=lambda h: f"{h.upper()} — {OPIS_HORYZONTU[h]}",
)

art = bundle["models"][wybrany_horyzont]
m = art["metrics"]

# ✅ INFO O KALIBRACJI
st.sidebar.markdown("---")
if art.get("calibrator") is not None:
    st.sidebar.success(f"✅ Model **skalibrowany** ({art.get('calibrator_type', 'isotonic')})")
else:
    st.sidebar.warning("⚠️ Model **nieskalibrowany** — score względny")

# Metryki
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Metryki modelu")
st.sidebar.metric("PR-AUC", f"{m['pr_auc']:.4f}",
                  help=f"lift {m['pr_auc_lift']:.1f}× baseline")
st.sidebar.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
st.sidebar.metric("Recall (Czułość)", f"{m['recall']:.4f}")
st.sidebar.metric("Precision (Precyzja)", f"{m['precision']:.4f}")
st.sidebar.metric("F2 (β=2)", f"{m['fbeta']:.4f}")

# ✅ BRIER SCORE (jeśli skalibrowany)
if art.get("brier_after") is not None:
    st.sidebar.metric(
        "Brier score",
        f"{art['brier_after']:.6f}",
        delta=f"{art.get('brier_improvement', 0):.6f}",
        delta_color="inverse",
        help="Im niższy, tym lepsza kalibracja"
    )

# Parametry
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Parametry")
st.sidebar.write(f"**Źródło:** `{art.get('source', 'v3.0')}`")
st.sidebar.write(f"**Liczba cech (top-N):** {art['top_n']}")
st.sidebar.write(f"**Bez flagi insolvency:** {art['drop_insolvency']}")
st.sidebar.write(f"**Użyto lagów:** {art['use_lags']}")
st.sidebar.write(f"**Próg decyzyjny:** {art['threshold']:.4f}")
st.sidebar.write(f"**PR-AUC lift:** {m['pr_auc_lift']:.1f}×")

if art.get("reliability"):
    st.sidebar.write(f"**Wiarygodność:** {ETYKIETA_WIARYGODNOSCI.get(art['reliability'], '—')}")

# ============================================================
# ZAKŁADKI
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predykcja", "📊 Statystyki", "🎯 Kalibracja", "📖 O modelach"
])

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

    domyslne = wczytaj_wartosci_domyslne(SCIEZKA_BUNDLE, wybrany_horyzont)

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
        proba, pred, is_calibrated = przewiduj(art, wartosci_wejsciowe)

        st.markdown("---")
        st.subheader("📊 Wynik predykcji")
        
        # ✅ INFO O KALIBRACJI
        if is_calibrated:
            st.success("✅ Prawdopodobieństwo **skalibrowane** (isotonic regression)")
        else:
            st.warning("⚠️ Score **nieskalibrowany** — traktuj jako względny ranking ryzyka")
        
        k1, k2, k3 = st.columns(3)
        with k1:
            label = "Prawdopodobieństwo" if is_calibrated else "Score ryzyka"
            st.metric(label, f"{proba*100:.2f}%")
        with k2:
            st.metric("Klasyfikacja",
                      "🔴 BANKRUCTWO" if pred else "🟢 BRAK BANKRUCTWA")
        with k3:
            st.metric("Próg", f"{art['threshold']:.4f}")

        st.progress(min(max(proba, 0.0), 1.0))

        if pred:
            st.error(
                f"⚠️ **Wysokie ryzyko bankructwa**\n\n"
                f"- {'Prawdopodobieństwo' if is_calibrated else 'Score ryzyka'}: "
                f"**{proba*100:.2f}%**\n"
                f"- Próg decyzyjny: **{art['threshold']:.4f}**\n"
                f"- Horyzont: **{OPIS_HORYZONTU[wybrany_horyzont]}**\n"
                f"- Wiarygodność modelu: "
                f"**{ETYKIETA_WIARYGODNOSCI.get(art.get('reliability'), '—')}**"
            )
        else:
            st.success(
                f"✅ **Niskie ryzyko bankructwa**\n\n"
                f"- {'Prawdopodobieństwo' if is_calibrated else 'Score ryzyka'}: "
                f"**{proba*100:.2f}%**\n"
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

# ---------- ZAKŁADKA 3: KALIBRACJA (NOWE) ----------
with tab3:
    st.header(f"🎯 Kalibracja — {wybrany_horyzont.upper()}")

    if art.get("calibrator") is None:
        st.warning(
            "⚠️ **Ten model nie jest skalibrowany.**\n\n"
            "Aby uzyskać skalibrowane prawdopodobieństwa:\n"
            "1. Uruchom `calibrate_models.py`\n"
            "2. Wybierz bundle **Skalibrowane** w panelu bocznym"
        )
    else:
        # Info o kalibracji
        st.success(f"✅ Model skalibrowany metodą **{art.get('calibrator_type', 'isotonic')}**")
        
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric(
                "Brier score (przed)",
                f"{art.get('brier_before', 0):.6f}",
            )
        with k2:
            st.metric(
                "Brier score (po)",
                f"{art.get('brier_after', 0):.6f}",
                delta=f"{art.get('brier_improvement', 0):.6f}",
                delta_color="inverse",
            )
        with k3:
            improvement_pct = (
                (art.get('brier_before', 0) - art.get('brier_after', 0)) 
                / max(art.get('brier_before', 1), 1e-8) * 100
            )
            st.metric(
                "Poprawa",
                f"{improvement_pct:.1f}%",
            )
        
        st.markdown("---")
        st.subheader("📊 Reliability Diagram")
        st.caption(
            "Diagram pokazuje, jak dobrze przewidywane prawdopodobieństwa "
            "odpowiadają rzeczywistym częstościom. Im bliżej przekątnej, tym lepiej."
        )
        
        # Oblicz reliability diagram
        try:
            # Załaduj dane testowe
            df_test = wczytaj_statystyki(wybrany_horyzont)
            
            # Przygotuj X
            drop_cols = ['company', 'link', 'industry', 'num', 'emis_id', 
                         'year', 'main_label']
            drop_cols = [c for c in drop_cols if c in df_test.columns]
            
            features = art["features"]
            medians = pd.Series(art["median_fill"])
            
            X_test = df_test.drop(columns=drop_cols).select_dtypes(include=[np.number])
            X_test = X_test[features]
            X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(medians)
            y_test = df_test["main_label"].values
            
            # Predykcje
            proba_raw = art["model"].predict_proba(X_test)[:, 1]
            proba_cal = art["calibrator"].predict(proba_raw)
            
            # Reliability diagram
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Przed kalibracją
            fraction_pos, mean_pred = calibration_curve(
                y_test, proba_raw, n_bins=10, strategy='quantile'
            )
            ax.plot(mean_pred, fraction_pos, 's--', 
                    label='Przed kalibracją', color='red', alpha=0.7)
            
            # Po kalibracji
            fraction_pos_cal, mean_pred_cal = calibration_curve(
                y_test, proba_cal, n_bins=10, strategy='quantile'
            )
            ax.plot(mean_pred_cal, fraction_pos_cal, 'o-', 
                    label='Po kalibracji', color='green', linewidth=2)
            
            # Diagonalna (ideał)
            ax.plot([0, 1], [0, 1], 'k--', label='Ideał', alpha=0.5)
            
            ax.set_xlabel('Średnie przewidywane prawdopodobieństwo', fontsize=11)
            ax.set_ylabel('Frakcja pozytywów (rzeczywista)', fontsize=11)
            ax.set_title(f'Reliability Diagram — {wybrany_horyzont.upper()}', 
                         fontsize=13, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            
            st.pyplot(fig)
            
            # Tabela z danymi
            st.markdown("### 📋 Dane kalibracji")
            df_cal = pd.DataFrame({
                "Przewidywane": mean_pred_cal,
                "Rzeczywiste": fraction_pos_cal,
                "Różnica": np.abs(mean_pred_cal - fraction_pos_cal),
            })
            st.dataframe(df_cal, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"❌ Błąd podczas tworzenia diagramu: {e}")

# ---------- ZAKŁADKA 4: O MODELACH ----------
with tab4:
    st.header("📖 O modelach")
    
    st.markdown(f"""
    **Bundle:** `{os.path.basename(SCIEZKA_BUNDLE)}`
    
    **Wersja:** {bundle.get('version', 'N/A')}
    
    **Metoda kalibracji:** {bundle.get('calibration_method', 'brak')}
    
    **Zasada wyboru:** dla każdego horyzontu wybrano najlepszy model 
    według PR-AUC lift.
    """)

    wiersze_wyboru = []
    for h in dostepne_horyzonty:
        a = bundle["models"][h]
        wiersze_wyboru.append({
            "Horyzont": h.upper(),
            "Top-N": a["top_n"],
            "Lagi": a.get("use_lags", False),
            "Bez flagi insolvency": a.get("drop_insolvency", False),
            "PR-AUC lift": a["metrics"]["pr_auc_lift"],
            "Skalibrowany": "✅" if a.get("calibrator") else "❌",
            "Brier (po)": a.get("brier_after", "—"),
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
    
    # Zbierz metryki
    metryki_rows = [
        ("PR-AUC", m["pr_auc"]),
        ("PR-AUC baseline", m["baseline_pr_auc"]),
        ("PR-AUC lift", m["pr_auc_lift"]),
        ("ROC-AUC", m["roc_auc"]),
        ("Recall (Czułość)", m["recall"]),
        ("Precision (Precyzja)", m["precision"]),
        ("F1", m["f1"]),
        ("F2 (β=2)", m["fbeta"]),
        ("Specificity (Swoistość)", m["specificity"]),
        ("NPV", m["npv"]),
        ("TP", m["TP"]),
        ("FP", m["FP"]),
        ("FN", m["FN"]),
        ("TN", m["TN"]),
    ]
    
    # Dodaj metryki kalibracji jeśli są
    if art.get("brier_before") is not None:
        metryki_rows.extend([
            ("Brier score (przed)", art["brier_before"]),
            ("Brier score (po)", art["brier_after"]),
            ("Brier improvement", art.get("brier_improvement", 0)),
        ])
    
    tabela_metryk = pd.DataFrame(metryki_rows, columns=["Metryka", "Wartość"])
    st.dataframe(tabela_metryk, use_container_width=True, hide_index=True)

# ---------- STOPKA ----------
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**🏦 Predykcja Bankructwa — Hybryda**\n\n"
    f"Bundle: `{os.path.basename(SCIEZKA_BUNDLE)}`  \n"
    f"Powered by Streamlit"
)