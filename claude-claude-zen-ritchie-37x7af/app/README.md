# 🛡️ Radar Upadłości — aplikacja do prezentacji

Aplikacja Streamlit, która na podstawie ~10 wskaźników finansowych ocenia ryzyko bankructwa firmy
w horyzontach **od 1 do 6 lat**, osobno dla trzech typów użytkownika:

| Profil | Cel | Jak wyznaczony próg |
|---|---|---|
| 🏦 Bank — kredyty | wyłapać ≥ 99% bankrutów | najwyższy próg z recall ≥ 99% na walidacji |
| ⚖️ Monitoring / audyt | balans | maksymalne F1 na walidacji |
| 📉 Fundusz — krótka sprzedaż | maksymalna trafność sygnału | maks. dolna granica 95% CI precyzji na walidacji |

Modele są trenowane tym samym pipeline'em co w `Project_01.ipynb`: podział czasowy train/val/test,
downsampling 1:5, RandomizedSearchCV, a próg wyznaczany na walidacji. Kolumny `Insolvency_flag`
i `operational_status` są **wykluczone**: to informacja „z przyszłości”. W poprzedniej wersji aplikacji
były w formularzu i to one dawały ROC-AUC 0,995.

## 1. Instalacja

```bash
cd app
pip install -r requirements.txt
```

## 2. Trening modeli (raz, na prawdziwych danych)

```bash
python train_models.py --data-dir ŚCIEŻKA/DO/V4FinBench_data_load-main
```

Skrypt potrzebuje plików `company_years_h1.parquet` … `company_years_h6.parquet`. Wybiera automatycznie
10 cech (średnia ważność ze wszystkich horyzontów, bez cech zdublowanych), trenuje 6 modeli
i zapisuje wszystko w `models/bundle.joblib`. Na wyjściu wypisuje wybrane cechy i wyniki każdego horyzontu.

Przydatne opcje:

- `--n-features 8`: liczba cech w formularzu,
- `--features "Kol1,Kol2,..."`: własna lista cech zamiast automatycznego wyboru,
- `--n-iter 30`: dokładniejszy tuning (dłużej trwa),
- `--bank-recall 0.99 --fund-min-recall 0.05`: parametry profili,
- `--demo`: dane syntetyczne, tylko do sprawdzenia, czy aplikacja działa (w aplikacji pojawia się wtedy pomarańczowa ramka „TRYB DEMO”).

Jeśli w danych jest cecha bez polskiego opisu, aplikacja przetłumaczy jej nazwę automatycznie.
Ładniejszy opis można dopisać w `feature_info.py`.

## 3. Uruchomienie

Lokalnie:

```bash
streamlit run app.py
```

Publicznie przez ngrok (link do wysłania jury):

```bash
export NGROK_AUTHTOKEN=twój_token        # https://dashboard.ngrok.com/get-started/your-authtoken
python start_ngrok.py                    # albo: python start_ngrok.py --auth demo:haslo123
```

Alternatywnie, bez Pythona: `streamlit run app.py`, a w drugim terminalu `ngrok http 8501`.

## 4. Co jest w aplikacji

1. **🔎 Ocena firmy**
   - suwaki pogrupowane w kategorie (płynność, zadłużenie, rentowność…) z objaśnieniami po polsku,
   - gotowe przykłady: archetypy („Rentowna, ale bez gotówki”, „Szybki wzrost na kredyt”…) oraz prawdziwe, anonimowe firmy ze zbioru testowego,
   - werdykt dla wybranego profilu, wskaźnik ryzyka 0–100 i prawdopodobieństwo upadłości skorygowane o downsampling,
   - **krzywa ryzyka 1–6 lat** z progiem alarmu,
   - **dlaczego?**: wkład każdej cechy (SHAP),
   - **firma na tle rynku**: porównanie z typową zdrową firmą i typowym bankrutem,
   - **co poprawić?**: 3 zmiany, które najbardziej obniżają ryzyko.
2. **📂 Portfel firm**: ocena 300 prawdziwych firm testowych albo własnego CSV (szablon do pobrania), z mini-wykresem ryzyka 1–6 lat dla każdej firmy.
3. **💼 Wartość biznesowa**:
   - bank: uniknięte straty kredytowe minus koszt weryfikacji, odsetek wniosków zatwierdzanych automatycznie,
   - fundusz: trafność vs próg opłacalności i oczekiwany zwrot na pozycję.
4. **🧪 Jakość modelu**: recall i precyzja każdego profilu w 6 horyzontach, ROC, PR, rozkład wyniku oraz suwak „zagraj progiem” pokazujący kompromis na 10 000 firm.
5. **ℹ️ Jak to działa**: metodologia i słowniczek cech.

## 5. Scenariusz pitchu (~5 minut)

1. **Problem (30 s).** Mniej niż 1 na 300 firm upada, ale każda upadłość to strata dla banku albo okazja dla funduszu. Jeden model nie obsłuży obu tych potrzeb.
2. **Demo firmy (90 s).**
   - Wczytaj „🟠 Rentowna, ale bez gotówki” i pokaż, że zysk to nie wszystko.
   - Pokaż wykres „Dlaczego?”, potem przesuń suwak płynności i obserwuj, jak gaśnie alarm.
3. **Czas (30 s).** Na krzywej ryzyka 1–6 lat pokaż, jak wcześnie pojawia się pierwszy sygnał.
4. **Dla kogo (60 s).**
   - Przełącz profil Bank → Fundusz: ta sama firma, inna decyzja.
   - Zakładka „Wartość biznesowa”: bank automatycznie zatwierdza X% wniosków, a fundusz ma trafność Y× lepszą od losowania.
5. **Wiarygodność (60 s).**
   - Zakładka „Jakość modelu”: test na latach, których model nie widział. Pokaż suwak progu.
   - Powiedz wprost, że usunęliśmy kolumny z przyszłości.
6. **Portfel (30 s).** 300 firm w sekundę, z eksportem do CSV.
