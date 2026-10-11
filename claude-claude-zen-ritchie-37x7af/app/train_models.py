"""Trenuje modele XGBoost dla horyzontów h1..h6 i zapisuje je w jednym pliku dla aplikacji.

Pipeline jest taki sam jak w notebooku Project_01 (część h2):
  * podział czasowy: train < VAL_YEAR, walidacja = VAL_YEAR, test > VAL_YEAR,
  * downsampling zbioru treningowego do proporcji 1 bankrut : 5 zdrowych,
  * RandomizedSearchCV dla XGBoost,
  * progi decyzyjne wyznaczane na WALIDACJI, ocena na TEŚCIE.
Różnice: wszystkie horyzonty używają tego samego, małego zestawu ~10 cech
(żeby w aplikacji był jeden formularz), a cechy wybieramy po średniej
ważności ze wszystkich horyzontów z odrzuceniem cech silnie skorelowanych.

Użycie:
    python train_models.py --data-dir ŚCIEŻKA_DO_PARQUETÓW
    python train_models.py --demo        # dane syntetyczne, tylko do podglądu aplikacji
"""

import argparse
import datetime as dt
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample

from feature_info import FEATURE_INFO, LEAKY_OR_ID_COLS

RATIO_TARGET = 5          # 1 bankrut : 5 zdrowych (jak w notebooku)
DEFAULT_VAL_YEAR = 2016
PARAM_GRID = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
}

PROFILES = {
    'bank': 'Bank — udzielanie kredytów (wyłap ≥ {bank_recall:.0%} bankrutów)',
    'balanced': 'Monitoring / audyt (najlepszy balans F1)',
    'fund': 'Fundusz — krótka sprzedaż (maksymalna precyzja)',
}


def horizon_file(k):
    return f'company_years_h{k}.parquet'


def find_data_dir(explicit):
    if explicit:
        return Path(explicit)
    found = list(Path.cwd().rglob(horizon_file(1))) + list(Path.cwd().parent.rglob(horizon_file(1)))
    if not found:
        raise SystemExit('Nie znaleziono company_years_h1.parquet — podaj --data-dir.')
    return found[0].parent


def choose_split(years, labels):
    """Domyślnie val = 2016, test > 2016. Jeśli w którymś zbiorze jest za mało bankrutów
    (np. dla długich horyzontów), przesuwamy podział: test = 2 ostatnie lata, val = rok wcześniej."""
    def ok(val_year):
        tr = labels[years < val_year].sum()
        va = labels[years == val_year].sum()
        te = labels[years > val_year].sum()
        return tr >= 50 and va >= 20 and te >= 30

    if ok(DEFAULT_VAL_YEAR):
        return DEFAULT_VAL_YEAR
    uniq = sorted(np.unique(years))
    for val_year in reversed(uniq[:-1]):
        if ok(val_year):
            return int(val_year)
    raise SystemExit('Za mało bankructw, żeby wydzielić train/val/test.')


def downsample(X, y, seed=42):
    pos = y == 1
    n_neg = min(int((~pos).sum()), int(pos.sum()) * RATIO_TARGET)
    neg_idx = resample(np.where(~pos)[0], replace=False, n_samples=n_neg, random_state=seed)
    idx = np.concatenate([neg_idx, np.where(pos)[0]])
    return X.iloc[idx], y.iloc[idx]


def candidate_columns(df):
    cols = [c for c in df.select_dtypes(include='number').columns if c not in LEAKY_OR_ID_COLS]
    # cechy z dużą liczbą braków są niewygodne w formularzu
    return [c for c in cols if df[c].isna().mean() < 0.5]


def select_features(data_dir, horizons, n_features):
    """Średnia ważność (gain) ze wszystkich horyzontów + odrzucanie cech skorelowanych > 0.8."""
    importances, samples = [], []
    for k in horizons:
        df = pd.read_parquet(data_dir / horizon_file(k))
        cols = candidate_columns(df)
        val_year = choose_split(df['year'].values, df['main_label'].values)
        tr = df[df['year'] < val_year]
        X_ds, y_ds = downsample(tr[cols], tr['main_label'])
        m = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                              importance_type='gain', random_state=42, n_jobs=-1)
        m.fit(X_ds, y_ds)
        imp = pd.Series(m.feature_importances_, index=cols)
        importances.append(imp / imp.sum())
        samples.append(X_ds)
        print(f'  h{k}: ranking cech gotowy ({len(cols)} kandydatów)')
        del df

    mean_imp = pd.concat(importances, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    pool = pd.concat(samples)[mean_imp.index]
    corr = pool.rank().corr().abs()

    chosen = []
    for col in mean_imp.index:
        if all(corr.loc[col, c] <= 0.8 for c in chosen):
            chosen.append(col)
        if len(chosen) == n_features:
            break
    print('\nWybrane cechy:')
    for c in chosen:
        print(f'  {mean_imp[c]:.3f}  {c}  ->  {FEATURE_INFO.get(c, {}).get("label", "(opis automatyczny)")}')
    return chosen


def pick_thresholds(y_val, s_val, bank_recall, fund_min_recall):
    p, r, t = precision_recall_curve(y_val, s_val)
    p, r = p[:-1], r[:-1]
    out = {}
    ok = np.where(r >= bank_recall)[0]
    out['bank'] = float(t[ok[-1]] if len(ok) else t[0])
    with np.errstate(invalid='ignore', divide='ignore'):
        f1 = np.nan_to_num(2 * p * r / (p + r))
    out['balanced'] = float(t[np.argmax(f1)])
    # Fundusz: maksymalizujemy DOLNĄ granicę 95% przedziału Wilsona dla precyzji.
    # Sama precyzja przy bardzo wysokim progu opiera się na kilku firmach i jest losowa;
    # dolna granica premiuje progi, które dają wysoką precyzję na sensownej liczbie sygnałów.
    n_pos = y_val.sum()
    tp = r * n_pos
    n_alerts = np.maximum(tp / np.maximum(p, 1e-12), 1)
    z = 1.96
    lb = (p + z**2 / (2 * n_alerts) - z * np.sqrt(p * (1 - p) / n_alerts + z**2 / (4 * n_alerts**2))) \
        / (1 + z**2 / n_alerts)
    lb[(r < fund_min_recall) | (tp < 10)] = -1
    # fundusz ma być najbardziej ostrożnym profilem: nigdy niższy próg niż profil zbalansowany
    out['fund'] = float(max(t[np.argmax(lb)], out['balanced']))
    return out


def metrics_at(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=thr, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                alert_rate=(tp + fp) / len(y), specificity=tn / max(tn + fp, 1),
                f1=2 * tp / max(2 * tp + fp + fn, 1))


def thin(*arrays, n=400):
    idx = np.unique(np.linspace(0, len(arrays[0]) - 1, n).astype(int))
    return [np.asarray(a)[idx].tolist() for a in arrays]


def train_horizon(data_dir, k, features, args):
    df = pd.read_parquet(data_dir / horizon_file(k), columns=features + ['year', 'main_label'])
    years, labels = df['year'].values, df['main_label'].values
    val_year = choose_split(years, labels)
    tr, va, te = df[df['year'] < val_year], df[df['year'] == val_year], df[df['year'] > val_year]
    X_ds, y_ds = downsample(tr[features], tr['main_label'])

    search = RandomizedSearchCV(xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
                                PARAM_GRID, n_iter=args.n_iter, scoring='roc_auc', cv=3,
                                random_state=42, n_jobs=-1)
    search.fit(X_ds, y_ds)
    model = xgb.XGBClassifier(**search.best_params_, random_state=42, eval_metric='logloss', n_jobs=-1)
    model.fit(X_ds, y_ds)

    s_val = model.predict_proba(va[features])[:, 1]
    s_test = model.predict_proba(te[features])[:, 1]
    y_val, y_test = va['main_label'].values, te['main_label'].values

    thresholds = pick_thresholds(y_val, s_val, args.bank_recall, args.fund_min_recall)
    profiles = {name: metrics_at(y_test, s_test, thr) for name, thr in thresholds.items()}

    # korekta prawdopodobieństwa po downsamplingu (prior correction):
    # model widział 1:5, a w rzeczywistości bankrutów jest ~0.3%
    base_rate = tr['main_label'].mean()
    odds_factor = (base_rate / (1 - base_rate)) / (y_ds.mean() / (1 - y_ds.mean()))

    p, r, t = precision_recall_curve(y_test, s_test)
    fpr, tpr, _ = roc_curve(y_test, s_test)
    bins = np.linspace(0, 1, 41)
    info = dict(
        horizon=k,
        booster=bytes(model.get_booster().save_raw('json')),
        best_params=search.best_params_,
        split=dict(train=f'< {val_year}', val=str(val_year), test=f'> {val_year}',
                   n_train=len(tr), n_train_ds=len(X_ds), n_val=len(va), n_test=len(te),
                   pos_val=int(y_val.sum()), pos_test=int(y_test.sum())),
        thresholds=thresholds,
        profiles=profiles,
        roc_auc=float(roc_auc_score(y_test, s_test)),
        pr_auc=float(average_precision_score(y_test, s_test)),
        base_rate_test=float(y_test.mean()),
        odds_factor=float(odds_factor),
        score_quantiles=np.quantile(s_test, np.linspace(0, 1, 1001)).tolist(),
        pr_curve=dict(zip(['precision', 'recall', 'threshold'], thin(p[:-1], r[:-1], t))),
        roc_curve=dict(zip(['fpr', 'tpr'], thin(fpr, tpr))),
        hist=dict(bins=bins.tolist(),
                  healthy=np.histogram(s_test[y_test == 0], bins=bins, density=True)[0].tolist(),
                  bankrupt=np.histogram(s_test[y_test == 1], bins=bins, density=True)[0].tolist()),
    )
    print(f'  h{k}: val={val_year}  ROC-AUC={info["roc_auc"]:.3f}  PR-AUC={info["pr_auc"]:.3f}  ' +
          '  '.join(f'{n}: R={m["recall"]:.2f} P={m["precision"]:.3f}' for n, m in profiles.items()))
    return info, (tr, te, s_test) if k == min(args.horizons) else None


def feature_stats(train, features):
    stats = {}
    for c in features:
        x = train[c].dropna()
        stats[c] = dict(
            grid=np.quantile(x, np.linspace(0, 1, 101)).tolist(),   # do liczenia percentyla
            median=float(x.median()),
            healthy_median=float(train.loc[train['main_label'] == 0, c].median()),
            bankrupt_median=float(train.loc[train['main_label'] == 1, c].median()),
        )
    return stats


def pick_examples(test, s_test, features, stats):
    """Kilka prawdziwych (anonimowych) firm ze zbioru testowego do prezentacji + mały portfel."""
    t = test[features + ['year', 'main_label']].copy()
    t['score'] = s_test
    for c in features:
        t[c] = t[c].fillna(stats[c]['median'])
    rng = np.random.default_rng(7)
    bad, good = t[t['main_label'] == 1], t[t['main_label'] == 0]
    picks = [
        ('Prawdziwa firma A — upadła (model ostrzegł)', bad[bad['score'] >= bad['score'].quantile(0.6)]),
        ('Prawdziwa firma B — upadła (sygnał umiarkowany)', bad[(bad['score'] > bad['score'].quantile(0.3)) &
                                                              (bad['score'] < bad['score'].quantile(0.5))]),
        ('Prawdziwa firma C — zdrowa', good[good['score'] <= good['score'].quantile(0.3)]),
        ('Prawdziwa firma D — zdrowa, ale z wysokim wynikiem', good[good['score'] >= good['score'].quantile(0.99)]),
    ]
    examples = {}
    for name, pool in picks:
        if len(pool):
            row = pool.iloc[rng.integers(len(pool))]
            examples[name] = dict(values={c: float(row[c]) for c in features},
                                  year=int(row['year']), bankrupt=int(row['main_label']))
    n_bad = min(len(bad), 40)
    portfolio = pd.concat([bad.sample(n_bad, random_state=1), good.sample(300 - n_bad, random_state=1)])
    portfolio = portfolio.sample(frac=1, random_state=2).reset_index(drop=True)
    portfolio.insert(0, 'firma', [f'Firma {i + 1:03d}' for i in range(len(portfolio))])
    return examples, portfolio[['firma', 'year', 'main_label'] + features]


def make_demo_data(out_dir, n_firms=6000, seed=0):
    """Syntetyczne dane z nazwami kolumn jak w V4FinBench — tylko do podglądu aplikacji."""
    rng = np.random.default_rng(seed)
    years = np.arange(2008, 2023)
    firm = np.repeat(np.arange(n_firms), len(years))
    year = np.tile(years, n_firms)
    n = len(firm)
    health = rng.normal(size=n_firms)[firm] + rng.normal(scale=0.5, size=n)
    cols = {'num': np.arange(n), 'emis_id': firm, 'year': year, 'company': 'x', 'industry': 'x', 'link': 'x',
            'country': rng.integers(0, 4, n)}
    for i, (c, info) in enumerate(FEATURE_INFO.items()):
        sign = {True: 1, False: -1, None: 0.3}[info['higher_is_better']]
        w = 0.3 + 0.7 * ((i * 37) % 10) / 10
        cols[c] = 1 + 0.5 * sign * w * health + rng.normal(scale=0.6, size=n)
        cols[c][rng.random(n) < 0.08] = np.nan
    base = pd.DataFrame(cols)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(1, 7):
        logit = -6.3 - (1.9 - 0.2 * k) * health + rng.normal(scale=0.4 + 0.15 * k, size=n)
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        d = base.copy()
        d['main_label'] = y
        d['Insolvency_flag'] = y          # celowo "przeciekająca" kolumna — musi zostać odrzucona
        d['operational_status'] = y
        d.to_parquet(out_dir / horizon_file(k))
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', help='katalog z company_years_h1..h6.parquet')
    ap.add_argument('--out', default=str(Path(__file__).parent / 'models' / 'bundle.joblib'))
    ap.add_argument('--horizons', default='1,2,3,4,5,6')
    ap.add_argument('--n-features', type=int, default=10)
    ap.add_argument('--features', help='własna lista cech oddzielona przecinkami (pomija automatyczny wybór)')
    ap.add_argument('--n-iter', type=int, default=15, help='liczba kombinacji w RandomizedSearchCV')
    ap.add_argument('--bank-recall', type=float, default=0.99)
    ap.add_argument('--fund-min-recall', type=float, default=0.05)
    ap.add_argument('--demo', action='store_true', help='użyj danych syntetycznych')
    args = ap.parse_args()
    args.horizons = [int(h) for h in args.horizons.split(',')]

    if args.demo:
        data_dir = make_demo_data(Path(tempfile.mkdtemp()) / 'demo')
        print(f'⚠️  Tryb DEMO — dane syntetyczne w {data_dir}')
    else:
        data_dir = find_data_dir(args.data_dir)
    print(f'📁 Dane: {data_dir}\n')

    if args.features:
        features = [c.strip() for c in args.features.split(',')]
        bad = set(features) & LEAKY_OR_ID_COLS
        if bad:
            raise SystemExit(f'Te kolumny to cel/identyfikatory i nie mogą być cechami: {bad}')
    else:
        print('🔍 Wybór cech...')
        features = select_features(data_dir, args.horizons, args.n_features)

    print('\n🚀 Trenowanie modeli...')
    horizons = {}
    for k in args.horizons:
        info, extra = train_horizon(data_dir, k, features, args)
        horizons[k] = info
        if extra:
            first_train, first_test, first_scores = extra

    stats = feature_stats(first_train, features)
    examples, portfolio = pick_examples(first_test, first_scores, features, stats)

    bundle = dict(
        created=dt.datetime.now().isoformat(timespec='minutes'),
        demo=args.demo,
        data_dir=str(data_dir),
        features=features,
        feature_stats=stats,
        horizons=horizons,
        profiles_desc={k: v.format(bank_recall=args.bank_recall) for k, v in PROFILES.items()},
        settings=dict(bank_recall=args.bank_recall, fund_min_recall=args.fund_min_recall,
                      ratio_target=RATIO_TARGET, n_iter=args.n_iter),
        examples=examples,
        portfolio=portfolio,
        xgboost_version=xgb.__version__,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    print(f'\n✅ Zapisano: {out}')


if __name__ == '__main__':
    main()
