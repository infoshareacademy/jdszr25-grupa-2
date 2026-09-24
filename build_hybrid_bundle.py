# ============================================================
# build_hybrid_bundle.py
# ------------------------------------------------------------
# Buduje jeden bundle (pkl) z modeli v2.0 i v3.0.
# Dla każdego horyzontu (h1–h6) wybiera najlepszy model
# według PR-AUC lift (auto_select) lub według ręcznego nadpisania.
#
# Uruchomienie:
#   python build_hybrid_bundle.py            # auto + manual override
#   python build_hybrid_bundle.py --auto     # tylko auto_select
#   python build_hybrid_bundle.py --manual   # tylko manual override
#   python build_hybrid_bundle.py --report   # dodatkowy raport porównawczy
# ============================================================

import os
import time
import argparse
from collections import OrderedDict

import joblib
import pandas as pd


# ============================================================
# ŚCIEŻKI
# ============================================================
KATALOG_PRZETWORZONY = os.path.join("data", "processed")

SCIEZKA_V2 = os.path.join(KATALOG_PRZETWORZONY, "models_bundle.pkl")
SCIEZKA_V3 = os.path.join(KATALOG_PRZETWORZONY, "models_bundle_v3.pkl")
SCIEZKA_WY = os.path.join(KATALOG_PRZETWORZONY, "models_bundle_hybrid.pkl")

HORYZONTY = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']


# ============================================================
# RĘCZNE NADPISANIA
# ------------------------------------------------------------
# Jeśli horyzont znajduje się w tym słowniku, auto_select jest
# pomijany i używany jest wskazany model.
#
# Format: 'hX': ('v2' | 'v3', 'klucz_w_bundle')
# Zostaw pusty dict {}, aby ZAWSZE używać auto_select.
# ============================================================
MANUAL_OVERRIDE = {
    # 'h1': ('v2', 'h1__no_flag'),
    # 'h6': ('v3', 'h6__base__no_flag'),
}


# ============================================================
# WCZYTANIE BUNDLE
# ============================================================
def wczytaj_lub_pusty(sciezka):
    """Wczytuje bundle, jeśli istnieje; w przeciwnym razie zwraca None."""
    if os.path.exists(sciezka):
        try:
            b = joblib.load(sciezka)
            print(f"  ✅ wczytano: {sciezka}")
            return b
        except Exception as e:
            print(f"  ❌ błąd wczytywania {sciezka}: {type(e).__name__}: {e}")
            return None
    print(f"  ⚠️  nie znaleziono: {sciezka}")
    return None


# ============================================================
# NORMALIZACJA ARTEFAKTU
# ------------------------------------------------------------
# Ujednolica artefakty z v2.0 i v3.0 do wspólnego formatu,
# który rozumie API (app_2.py).
# ============================================================
def normalizuj_artefakt(art, znacznik_zrodla, klucz):
    """Zwraca znormalizowany artefakt gotowy do bundle hybrydowego."""
    m = art.get("metrics", {})

    def _f(name, default=0.0):
        v = m.get(name, default)
        return float(v) if v is not None else float(default)

    def _i(name, default=0):
        v = m.get(name, default)
        return int(v) if v is not None else int(default)

    return {
        # --- identyfikacja ---
        "horizon": art.get("horizon", klucz.split("__")[0]),
        "top_n": int(art.get("top_n", 0)),
        "source": znacznik_zrodla,                 # "v2" | "v3"
        "source_key": klucz,
        "drop_insolvency": bool(art.get("drop_insolvency", False)),
        "use_lags": bool(art.get("use_lags", False)),

        # --- model i preprocessing ---
        "model": art["model"],                     # xgb.XGBClassifier
        "features": list(art.get("features", [])),
        "threshold": float(art.get("threshold", 0.5)),
        "threshold_source": art.get("threshold_source", "val"),
        "median_fill": dict(art.get("median_fill", {})),

        # --- metryki (ujednolicone) ---
        "metrics": {
            "recall": _f("recall"),
            "precision": _f("precision"),
            "f1": _f("f1"),
            "fbeta": _f("fbeta", 0.0),
            "beta": _f("beta", 2.0),
            "roc_auc": _f("roc_auc"),
            "pr_auc": _f("pr_auc"),
            "baseline_pr_auc": _f("baseline_pr_auc"),
            "pr_auc_lift": _f("pr_auc_lift"),
            "specificity": _f("specificity"),
            "npv": _f("npv"),
            "TP": _i("TP"),
            "FP": _i("FP"),
            "FN": _i("FN"),
            "TN": _i("TN"),
            "n_bank_test": _i("n_bank_test"),
            "n_bank_test_with_flag0": m.get("n_bank_test_with_flag0"),
        },

        # --- parametry techniczne ---
        "n_features_numeric": int(art.get("n_features_numeric", 0)),
        "n_features_after_corr": int(art.get("n_features_after_corr", 0)),
        "ratio": float(art.get("ratio", 0.0)),
        "training_time_sec": float(art.get("training_time_sec", 0.0)),
    }


# ============================================================
# OCENA WIARYGODNOŚCI
# ============================================================
def okresl_wiarygodnosc(pr_auc_lift):
    """Etykieta wiarygodności na podstawie PR-AUC lift."""
    if pr_auc_lift >= 30:
        return "high"
    if pr_auc_lift >= 10:
        return "medium"
    return "low"


# ============================================================
# AUTO_SELECT
# ------------------------------------------------------------
# Spośród wszystkich bundle'ów i kluczy z danym horyzontem
# wybiera model o najwyższym PR-AUC lift.
# ============================================================
def auto_select(h, bundles):
    """Zwraca dict: {tag, key, lift, art} lub None."""
    best = None
    for tag, b in bundles.items():
        if b is None or "models" not in b:
            continue
        for key, art in b["models"].items():
            if not key.startswith(f"{h}__"):
                continue
            lift = art.get("metrics", {}).get("pr_auc_lift", 0.0) or 0.0
            if best is None or lift > best["lift"]:
                best = {"tag": tag, "key": key, "lift": lift, "art": art}
    return best


# ============================================================
# WYBÓR MODELU — auto + manual + fallback
# ============================================================
def wybierz_model(h, bundles, tryb="auto"):
    """
    Zwraca (tag, key, art).
    tryb:
        "auto"   — auto_select, potem fallback
        "manual" — tylko MANUAL_OVERRIDE, potem fallback
    """
    # 1) Ręczne nadpisanie (zawsze sprawdzane pierwsze)
    if h in MANUAL_OVERRIDE:
        tag, key = MANUAL_OVERRIDE[h]
        b = bundles.get(tag)
        if b and key in b.get("models", {}):
            print(f"  🔧 {h}: ręczne nadpisanie → {tag}/{key}")
            return tag, key, b["models"][key]
        print(f"  ⚠️  {h}: ręczne nadpisanie '{tag}/{key}' niedostępne, "
              f"przechodzę dalej")

    # 2) Automatyczny wybór
    if tryb == "auto":
        best = auto_select(h, bundles)
        if best is not None:
            print(f"  🤖 {h}: auto_select → {best['tag']}/{best['key']} "
                  f"(lift={best['lift']:.1f}x)")
            return best["tag"], best["key"], best["art"]

    # 3) Fallback: pierwszy klucz z danym h
    for tag, b in bundles.items():
        if b is None or "models" not in b:
            continue
        for key in b["models"]:
            if key.startswith(f"{h}__"):
                print(f"  ⚠️  {h}: fallback → {tag}/{key}")
                return tag, key, b["models"][key]

    print(f"  ❌ {h}: brak modelu w żadnym bundle")
    return None, None, None


# ============================================================
# RAPORT PORÓWNAWCZY
# ------------------------------------------------------------
# Pokazuje wszystkie dostępne modele per horyzont:
# który został wybrany, a które odrzucone i o ile.
# ============================================================
def raport_porownawczy(bundles, hybrid):
    """Wypisuje tabelę wszystkich kandydatów dla każdego horyzontu."""
    print("\n" + "=" * 100)
    print("📊 RAPORT PORÓWNAWCZY — wszyscy kandydaci per horyzont")
    print("=" * 100)

    for h in HORYZONTY:
        wiersze = []
        for tag, b in bundles.items():
            if b is None or "models" not in b:
                continue
            for key, art in b["models"].items():
                if not key.startswith(f"{h}__"):
                    continue
                m = art.get("metrics", {})
                wiersze.append({
                    "źródło": tag,
                    "klucz": key,
                    "top_n": art.get("top_n", 0),
                    "lagi": art.get("use_lags", False),
                    "bez_flagi": art.get("drop_insolvency", False),
                    "pr_auc": m.get("pr_auc", 0.0),
                    "lift": m.get("pr_auc_lift", 0.0),
                    "roc_auc": m.get("roc_auc", 0.0),
                    "recall": m.get("recall", 0.0),
                    "precision": m.get("precision", 0.0),
                })

        if not wiersze:
            print(f"\n{h.upper()}: brak kandydatów")
            continue

        df = pd.DataFrame(wiersze).sort_values("lift", ascending=False)
        wybrany_klucz = (
            hybrid["models"][h]["source_key"]
            if h in hybrid["models"] else None
        )
        df["✓"] = df["klucz"].apply(lambda k: "✅" if k == wybrany_klucz else "")

        print(f"\n{h.upper()} — wybrany: {wybrany_klucz}")
        print(df.to_string(index=False))


# ============================================================
# GŁÓWNA FUNKCJA
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Buduje hybrydowy bundle modeli (v2.0 + v3.0)"
    )
    parser.add_argument("--auto", action="store_true",
                        help="Użyj tylko auto_select (pomija MANUAL_OVERRIDE)")
    parser.add_argument("--manual", action="store_true",
                        help="Użyj tylko MANUAL_OVERRIDE")
    parser.add_argument("--report", action="store_true",
                        help="Wypisz raport porównawczy wszystkich kandydatów")
    args = parser.parse_args()

    # Tryb: manual ma priorytet, jeśli podano
    tryb = "manual" if args.manual else "auto"

    print("=" * 100)
    print(f"📦 Budowanie hybrydowego bundle  (tryb: {tryb})")
    print("=" * 100)

    # --- Wczytanie bundle ---
    bundles = OrderedDict()
    bundles["v2"] = wczytaj_lub_pusty(SCIEZKA_V2)
    bundles["v3"] = wczytaj_lub_pusty(SCIEZKA_V3)

    if all(b is None for b in bundles.values()):
        raise SystemExit(
            "❌ Brak bundle v2.0 i v3.0 — najpierw wytrenuj modele"
        )

    # --- Struktura hybrydy ---
    hybrid = {
        "version": "hybrid-1.0",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "horizons": HORYZONTY,
        "sources": {
            tag: os.path.basename(path)
            for tag, path in [("v2", SCIEZKA_V2), ("v3", SCIEZKA_V3)]
        },
        "auto_selection": (tryb == "auto"),
        "manual_override": dict(MANUAL_OVERRIDE),
        "selection": {},
        "models": {},
    }

    # --- Wybór modeli ---
    print("\n" + "-" * 100)
    print("🔍 Wybór modeli per horyzont")
    print("-" * 100)

    for h in HORYZONTY:
        tag, key, art = wybierz_model(h, bundles, tryb=tryb)
        if art is None:
            continue

        norm = normalizuj_artefakt(art, tag, key)
        norm["reliability"] = okresl_wiarygodnosc(
            norm["metrics"]["pr_auc_lift"]
        )

        hybrid["models"][h] = norm
        hybrid["selection"][h] = {
            "source": tag,
            "key": key,
            "pr_auc_lift": norm["metrics"]["pr_auc_lift"],
            "reliability": norm["reliability"],
        }

        m = norm["metrics"]
        print(f"  ✅ {h}: {tag}/{key}  "
              f"PR-AUC={m['pr_auc']:.4f} (lift={m['pr_auc_lift']:.1f}x)  "
              f"ROC={m['roc_auc']:.4f}  "
              f"REC={m['recall']:.4f}  "
              f"PREC={m['precision']:.4f}  "
              f"wiarygodność={norm['reliability']}")

    if not hybrid["models"]:
        raise SystemExit("❌ Nie udało się wybrać żadnego modelu")

    # --- Zapis bundle ---
    joblib.dump(hybrid, SCIEZKA_WY, compress=3)
    rozmiar_mb = os.path.getsize(SCIEZKA_WY) / 1e6
    print(f"\n💾 Zapisano: {SCIEZKA_WY}  ({rozmiar_mb:.2f} MB)")

    # --- Podsumowanie ---
    print("\n" + "=" * 100)
    print("🏆 BUNDLE HYBRYDOWY — podsumowanie")
    print("=" * 100)
    wiersze = []
    for h, art in hybrid["models"].items():
        m = art["metrics"]
        wiersze.append({
            "horyzont": h,
            "źródło": art["source"],
            "klucz": art["source_key"],
            "top_n": art["top_n"],
            "lagi": art["use_lags"],
            "bez_flagi": art["drop_insolvency"],
            "recall": round(m["recall"], 4),
            "precision": round(m["precision"], 4),
            "f1": round(m["f1"], 4),
            "f2": round(m["fbeta"], 4),
            "pr_auc": round(m["pr_auc"], 4),
            "lift": round(m["pr_auc_lift"], 1),
            "roc_auc": round(m["roc_auc"], 4),
            "wiarygodność": art["reliability"],
        })
    df_sum = pd.DataFrame(wiersze)
    print(df_sum.to_string(index=False))

    # --- Opcjonalny raport porównawczy ---
    if args.report:
        raport_porownawczy(bundles, hybrid)

    print("\n✅ Gotowe. Uruchom aplikację:")
    print("   python -m streamlit run app_2.py")


# ============================================================
# START
# ============================================================
if __name__ == "__main__":
    main()