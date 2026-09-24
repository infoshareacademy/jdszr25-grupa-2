# ============================================================
# train_grid_v3.py — grid search dla h1..h6 z lagami i bez
# ------------------------------------------------------------
# Dla każdego horyzontu (h1–h6) i każdej konfiguracji:
#   • base (bez lagów) vs lag (z lagami)
#   • with_flag vs no_flag (drop_insolvency)
# trenuje modele XGBoost dla TOP_N ∈ {20..50}.
#
# Wybór najlepszego modelu: max PR-AUC → tie F2 → tie recall.
# Próg decyzyjny: F2-score (β=2), dobierany na wewnętrznym val.
#
# Wynik:
#   data/processed/models_bundle_v3.pkl        — bundle z modelami v3
#   data/processed/grid_topN_results_v3.csv    — pełny grid wyników
#   data/processed/best_per_horizon_v3.csv     — podsumowanie
# ============================================================

import os, gc, time, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)


# ============================================================
# KONFIGURACJA
# ============================================================
KATALOG_DANYCH    = "data"
KATALOG_PANEL     = os.path.join(KATALOG_DANYCH, "panel")
KATALOG_PRZETWORZONY = os.path.join(KATALOG_DANYCH, "processed")
os.makedirs(KATALOG_PRZETWORZONY, exist_ok=True)

HORYZONTY      = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
TOP_N_RANGE    = [20, 25, 30, 35, 40, 45, 50]
CORR_THRESHOLD = 0.8

# F_beta: beta=2 → recall waży 2× silniej niż precision
BETA = 2.0

# Próg decyzyjny dobierany na wewnętrznym zbiorze walidacyjnym
USE_VAL_FOR_THRESHOLD = True

XGB_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, reg_alpha=0.1, reg_lambda=2,
    random_state=42, eval_metric='aucpr', n_jobs=-1,
)


# ============================================================
# FUNKCJE POMOCNICZE
# ============================================================
def _fbeta(p, r, beta=BETA):
    """F_beta score. Bezpieczne dla p+r == 0."""
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)


def _pick_threshold(y_true, y_proba, beta=BETA):
    """Dobiera próg maksymalizujący F_beta na siatce 0.01–0.99."""
    best = {'fbeta': -1.0, 'thresh': 0.5,
            'prec': 0.0, 'rec': 0.0, 'f1': 0.0,
            'fallback': True}
    for thresh in np.arange(0.01, 0.99, 0.005):
        y_pred = (y_proba >= thresh).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fb = _fbeta(p, r, beta)
        if fb > best['fbeta']:
            best = {'fbeta': float(fb), 'thresh': float(thresh),
                    'prec': float(p), 'rec': float(r), 'f1': float(f1),
                    'fallback': False}
    return best


# ============================================================
# BUDOWA MODELU
# ============================================================
def build_model(h, top_n, drop_insolvency=False, use_lags=False):
    """
    Buduje model XGBoost dla (h, top_n, drop_insolvency, use_lags).

    Kroki:
      1. Wczytanie parquet (base lub lagged)
      2. Usunięcie kolumn tekstowych i ID
      3. Wybór kolumn numerycznych
      4. Konwersja inf → NaN
      5. Podział StratifiedGroupKFold (po emis_id)
      6. Imputacja medianami z TRAIN
      7. Usunięcie cech o korelacji > CORR_THRESHOLD
      8. Selekcja top_n cech (XGBoost feature importance)
      9. Trening finalnego modelu na pełnym TRAIN
     10. Dobór progu (val lub test)
     11. Predykcja na TEST + metryki
     12. Zapis artefaktu (model, features, threshold, metrics)
    """
    t_start = time.time()

    # ---- 1. Wczytanie danych ----
    if use_lags:
        path = os.path.join(KATALOG_PANEL, f"company_years_{h}_lagged.parquet")
    else:
        path = os.path.join(KATALOG_DANYCH, f"company_years_{h}.parquet")

    if not os.path.exists(path):
        return {"skipped": True, "reason": "file_missing",
                "horizon": h, "top_n": top_n,
                "drop_insolvency": drop_insolvency, "use_lags": use_lags}

    df = pd.read_parquet(path)
    df = df.dropna(subset=['main_label']).copy()
    df['main_label'] = df['main_label'].astype(int)

    # ---- 2. ID + drop kolumn tekstowych ----
    company_id = df['emis_id'].values
    columns_to_drop = [c for c in ['company','link','industry','num','emis_id','year']
                       if c in df.columns]
    if drop_insolvency and 'Insolvency_flag' in df.columns:
        columns_to_drop.append('Insolvency_flag')
    df_clean = df.drop(columns=columns_to_drop)

    # ---- 3. Tylko numeryczne ----
    X = df_clean.drop(columns=['main_label']).select_dtypes(include=[np.number])
    y = df_clean['main_label'].values
    assert len(company_id) == len(X)

    # ---- 4. inf → NaN ----
    X = X.replace([np.inf, -np.inf], np.nan)

    # ---- 5. Podział train/test ----
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups=company_id))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]
    g_train = company_id[train_idx]

    n_pos = int((y_train == 1).sum())
    if n_pos < 2:
        return {"skipped": True, "reason": "too_few_positives_train",
                "horizon": h, "top_n": top_n,
                "drop_insolvency": drop_insolvency, "use_lags": use_lags}
    ratio = float((y_train == 0).sum() / n_pos)

    # ---- 6. Imputacja medianami z TRAIN ----
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test  = X_test.fillna(medians)

    # ---- 7. Usunięcie skorelowanych cech ----
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESHOLD)]
    X_train_uncorr = X_train.drop(columns=to_drop)
    X_test_uncorr  = X_test.drop(columns=to_drop)

    if top_n > X_train_uncorr.shape[1]:
        return {"skipped": True, "reason": "not_enough_features",
                "n_after_corr": int(X_train_uncorr.shape[1]),
                "horizon": h, "top_n": top_n,
                "drop_insolvency": drop_insolvency, "use_lags": use_lags}

    # ---- 8. Selekcja cech ----
    selector = xgb.XGBClassifier(scale_pos_weight=ratio, **XGB_PARAMS)
    selector.fit(X_train_uncorr, y_train)
    imp = (pd.DataFrame({'feature': X_train_uncorr.columns,
                         'importance': selector.feature_importances_})
             .sort_values('importance', ascending=False))
    top_features = imp.head(top_n)['feature'].tolist()

    X_train_top = X_train_uncorr[top_features]
    X_test_top  = X_test_uncorr[top_features]

    # ---- 9. Dobór progu (val) ----
    if USE_VAL_FOR_THRESHOLD:
        sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
        tr2_idx, val_idx = next(sgkf_val.split(X_train_top, y_train, groups=g_train))

        X_tr2 = X_train_top.iloc[tr2_idx].reset_index(drop=True)
        X_val = X_train_top.iloc[val_idx].reset_index(drop=True)
        y_tr2, y_val = y_train[tr2_idx], y_train[val_idx]

        m_th = xgb.XGBClassifier(scale_pos_weight=ratio, **XGB_PARAMS)
        m_th.fit(X_tr2, y_tr2)
        proba_val = m_th.predict_proba(X_val)[:, 1]

        if (y_val == 1).sum() >= 2:
            best = _pick_threshold(y_val, proba_val, BETA)
        else:
            # Fallback: dobór progu na train (gdy zbyt mało pozytywów w val)
            proba_tr = m_th.predict_proba(X_tr2)[:, 1]
            best = _pick_threshold(y_tr2, proba_tr, BETA)
        thresh_source = "val"
    else:
        best = None
        thresh_source = "test"

    # ---- 10. Finalny model na pełnym TRAIN ----
    model = xgb.XGBClassifier(scale_pos_weight=ratio, **XGB_PARAMS)
    model.fit(X_train_top, y_train)

    # ---- 11. Predykcja na TEST ----
    y_proba = model.predict_proba(X_test_top)[:, 1]
    roc = float(roc_auc_score(y_test, y_proba))
    pr  = float(average_precision_score(y_test, y_proba))
    baseline_pr = float((y_test == 1).mean())

    if not USE_VAL_FOR_THRESHOLD:
        best = _pick_threshold(y_test, y_proba, BETA)

    if best['fallback']:
        return {"skipped": True, "reason": "no_valid_threshold",
                "horizon": h, "top_n": top_n,
                "drop_insolvency": drop_insolvency, "use_lags": use_lags}

    # ---- 12. Metryki ----
    thresh = best['thresh']
    y_pred = (y_proba >= thresh).astype(int)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    fb   = _fbeta(prec, rec, BETA)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv         = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    n_bank_total = int((y_test == 1).sum())
    n_bank_flag0 = None
    if 'Insolvency_flag' in df.columns:
        flag_test = df.loc[test_idx, 'Insolvency_flag'].values
        n_bank_flag0 = int(((y_test == 1) & (flag_test == 0)).sum())

    elapsed = time.time() - t_start

    # ---- 13. Artefakt ----
    artifact = {
        "horizon": h, "top_n": top_n,
        "drop_insolvency": bool(drop_insolvency),
        "use_lags": bool(use_lags),
        "model": model,
        "features": top_features,
        "threshold": float(thresh),
        "threshold_source": thresh_source,
        "median_fill": medians.to_dict(),
        "metrics": {
            "recall": float(rec), "precision": float(prec),
            "f1": float(f1), "fbeta": float(fb), "beta": BETA,
            "roc_auc": roc, "pr_auc": pr,
            "baseline_pr_auc": baseline_pr,
            "pr_auc_lift": float(pr / baseline_pr) if baseline_pr > 0 else None,
            "specificity": specificity, "npv": npv,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "n_bank_test": n_bank_total,
            "n_bank_test_with_flag0": n_bank_flag0,
        },
        "n_features_numeric": int(X.shape[1]),
        "n_features_after_corr": int(X_train_uncorr.shape[1]),
        "ratio": ratio,
        "training_time_sec": float(elapsed),
    }
    return artifact


# ============================================================
# GŁÓWNA PĘTLA
# ============================================================
if __name__ == "__main__":
    # Konfiguracje: jakie warianty liczymy dla każdego horyzontu
    CONFIGS = {
        'h1': [dict(drop_insolvency=False, use_lags=False),
               dict(drop_insolvency=False, use_lags=True)],
        'h2': [dict(drop_insolvency=False, use_lags=False),
               dict(drop_insolvency=False, use_lags=True)],
        'h3': [dict(drop_insolvency=False, use_lags=False),
               dict(drop_insolvency=False, use_lags=True)],
        'h4': [dict(drop_insolvency=False, use_lags=False),
               dict(drop_insolvency=False, use_lags=True)],
        'h5': [dict(drop_insolvency=False, use_lags=False),
               dict(drop_insolvency=False, use_lags=True)],
        'h6': [dict(drop_insolvency=False, use_lags=False),
               dict(drop_insolvency=False, use_lags=True)],
    }

    all_results = []
    bundle = {
        "version": "3.0",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "horizons": HORYZONTY,
        "top_n_range": TOP_N_RANGE,
        "beta": BETA,
        "corr_threshold": CORR_THRESHOLD,
        "use_val_for_threshold": USE_VAL_FOR_THRESHOLD,
        "xgboost_params": XGB_PARAMS,
        "models": {},
        "grid": [],
    }

    for h, configs in CONFIGS.items():
        for cfg in configs:
            drop_ins = cfg["drop_insolvency"]
            use_lags = cfg["use_lags"]
            key = f"{h}__{'lag' if use_lags else 'base'}__" \
                  f"{'no_flag' if drop_ins else 'with_flag'}"

            print("\n" + "="*100)
            print(f"🎯 {h.upper()} | {key}")
            print("="*100)

            artifacts = []
            rows = []
            for top_n in TOP_N_RANGE:
                try:
                    art = build_model(h, top_n,
                                      drop_insolvency=drop_ins,
                                      use_lags=use_lags)
                except Exception as e:
                    print(f"  top_n={top_n:2d} | ❌ {type(e).__name__}: {e}")
                    continue

                if art is None or art.get("skipped"):
                    reason = (art or {}).get("reason", "unknown")
                    print(f"  top_n={top_n:2d} | ⏭️  {reason}")
                    continue

                artifacts.append(art)
                m = art["metrics"]
                rows.append({
                    "horizon": h, "drop_insolvency": drop_ins,
                    "use_lags": use_lags,
                    "top_n": top_n, "threshold": art["threshold"],
                    "threshold_source": art["threshold_source"],
                    **m,
                    "n_features_after_corr": art["n_features_after_corr"],
                    "training_time_sec": art["training_time_sec"],
                })
                print(f"  top_n={top_n:2d} | "
                      f"REC={m['recall']:.4f} PREC={m['precision']:.4f} "
                      f"F2={m['fbeta']:.4f} PR-AUC={m['pr_auc']:.4f} "
                      f"(lift={m['pr_auc_lift']:.1f}x) "
                      f"ROC={m['roc_auc']:.4f} "
                      f"TP/FP/FN={m['TP']}/{m['FP']}/{m['FN']}")

            all_results.extend(rows)

            if not artifacts:
                print(f"  ⚠️  brak ważnych modeli")
                continue

            # Wybór najlepszego modelu: max PR-AUC → tie F2 → tie recall
            best = max(artifacts, key=lambda a: (
                a["metrics"]["pr_auc"],
                a["metrics"]["fbeta"],
                a["metrics"]["recall"],
            ))
            bundle["models"][key] = best
            bundle["grid"].extend(rows)

            m = best["metrics"]
            print(f"\n  🏆 BEST {key}: top_n={best['top_n']} | "
                  f"PR-AUC={m['pr_auc']:.4f} (lift={m['pr_auc_lift']:.1f}x) "
                  f"REC={m['recall']:.4f} PREC={m['precision']:.4f} "
                  f"F2={m['fbeta']:.4f}")

    # ---------------- ZAPIS ----------------
    res_df = pd.DataFrame(all_results)
    grid_csv = os.path.join(KATALOG_PRZETWORZONY, "grid_topN_results_v3.csv")
    res_df.to_csv(grid_csv, index=False)
    print(f"\n📄 Grid: {grid_csv}  ({len(res_df)} wierszy)")

    bundle_path = os.path.join(KATALOG_PRZETWORZONY, "models_bundle_v3.pkl")
    joblib.dump(bundle, bundle_path, compress=3)
    size_mb = os.path.getsize(bundle_path) / 1e6
    print(f"💾 Bundle: {bundle_path}  ({size_mb:.2f} MB)")

    # ---------------- PODSUMOWANIE ----------------
    print("\n" + "="*110)
    print("🏆 NAJLEPSZE MODELE (wybór po PR-AUC)")
    print("="*110)
    summary = []
    for key, art in bundle["models"].items():
        m = art["metrics"]
        summary.append({
            "key": key, "top_n": art["top_n"],
            "recall": m["recall"], "precision": m["precision"],
            "f1": m["f1"], "f2": m["fbeta"],
            "pr_auc": m["pr_auc"], "pr_auc_lift": m["pr_auc_lift"],
            "roc_auc": m["roc_auc"],
            "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
        })
    summary_df = pd.DataFrame(summary).sort_values("key")
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(KATALOG_PRZETWORZONY, "best_per_horizon_v3.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📄 Podsumowanie: {summary_path}")

    # ---------------- PORÓWNANIE base vs lag ----------------
    print("\n" + "="*110)
    print("📊 EFEKT LAGÓW: base vs lag (PR-AUC lift)")
    print("="*110)
    for h in HORYZONTY:
        base_key = f"{h}__base__with_flag"
        lag_key  = f"{h}__lag__with_flag"
        if base_key in bundle["models"] and lag_key in bundle["models"]:
            base_lift = bundle["models"][base_key]["metrics"]["pr_auc_lift"]
            lag_lift  = bundle["models"][lag_key]["metrics"]["pr_auc_lift"]
            delta = (lag_lift - base_lift) / base_lift * 100
            base_auc = bundle["models"][base_key]["metrics"]["pr_auc"]
            lag_auc  = bundle["models"][lag_key]["metrics"]["pr_auc"]
            print(f"  {h}: lift {base_lift:6.1f}x → {lag_lift:6.1f}x "
                  f"({delta:+6.1f}%) | "
                  f"PR-AUC {base_auc:.4f} → {lag_auc:.4f}")