# Stock Volume Dynamics – Analiza i prognozowanie wolumenu obrotu giełdowego

Projekt akademicki realizowany w ramach przedmiotu z zakresu analizy danych. Aplikacja realizuje pełny potok przetwarzania danych (ang. *end-to-end data pipeline*): od pobrania historycznych notowań giełdowych przez API, przez czyszczenie i składowanie w relacyjnej bazie danych, po inżynierię cech, modelowanie statystyczne/uczenie maszynowe oraz prezentację wyników w interaktywnym dashboardzie.

---

## 1. Cel i zakres projektu

Głównym celem projektu jest analiza dynamiki wolumenu obrotu akcjami (ang. *trading volume*) dla wybranej grupy spółek oraz budowa modelu prognozującego wolumen w określonych horyzontach czasowych.

Wolumen obrotu jest kluczową zmienną rynkową – niesie informacje o płynności waloru, natężeniu aktywności inwestorów oraz sile trendów cenowych. Projekt odpowiada na następujące pytania badawcze:
- Jak kształtuje się rozkład i sezonowość (dzienna, tygodniowa, miesięczna) wolumenu dla badanych spółek?
- W jakim stopniu przeszłe notowania cenowe i wskaźniki zmienności determinują przyszły wolumen obrotu?
- Z jaką dokładnością proste modele regresyjne i metody uczenia maszynowego są w stanie prognozować skoki wolumenu?

---

## 2. Architektura systemu i etapy realizacji

Zgodnie z wymaganiami projektowymi ([PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md)), system składa się z sześciu powiązanych etapów:

```text
[Zewnętrzne API]
       │
       ▼
1. Pobranie danych (skrypt zasilający)
       │
       ▼
2. Czyszczenie i transformacja (Preprocessing & EDA)
       │
       ▼
3. Relacyjna baza danych (SQLite)
       │
       ├──► 4. Eksport / zasilenie modułu modelowania
       │           │
       │           ▼
       │      5. Modelowanie predykcyjne (inżynieria cech, trenowanie, metryki)
       │           │
       │           ▼
       └──► 6. Zasilenie interaktywnego dashboardu (Streamlit)
```

### Opis modułów:
1. **Pozyskanie danych (API):**
   - Pobieranie danych historycznych (OHLCV: Open, High, Low, Close, Volume) z bezpłatnego interfejsu API (np. Yahoo Finance / Alpha Vantage / Tiingo).
   - Parametry zapytań (lista spółek, zakres dat, interwał) definiowane w zewnętrznym pliku konfiguracyjnym JSON.
2. **Czyszczenie i preprocessing:**
   - Weryfikacja spójności typów danych i ciągłości kalendarza sesji giełdowych.
   - Obsługa braków danych (np. dni wolne od handlu) oraz identyfikacja wartości odstających (anomalie wolumenowe).
   - Podstawowa eksploracja statystyczna (EDA).
3. **Baza danych (SQLite):**
   - Wykorzystanie wbudowanej bazy relacyjnej `SQLite`.
   - Zdefiniowane tabele na surowe notowania, przetworzone szeregi czasowe oraz wyniki ewaluacji modeli.
   - Relacyjna struktura zapewnia integralność danych pomiędzy spółkami a ich notowaniami.
4. **Wykorzystanie bazy w potoku:**
   - Baza nie jest pasywnym archiwum – skrypty modelowania odczytują z niej dane treningowe, a aplikacja webowa pobiera z niej aktualny stan danych i wygenerowane predykcje.
5. **Modelowanie predykcyjne:**
   - Konstrukcja cech (ang. *feature engineering*): opóźnienia czasowe (ang. *lags*), średnie kroczące (SMA/EMA wolumenu i ceny), zmienność (ang. *volatility*), rozpiętość świecy (High - Low), stopy zwrotu.
   - Implementacja modelu uczenia maszynowego (np. regresja liniowa/grzbietowa, Random Forest lub Gradient Boosting) do predykcji wolumenu na kolejny okres (np. $t+1$, $t+5$).
   - Ewaluacja na zbiorze testowym przy użyciu metryk: MAE, RMSE, MAPE oraz $R^2$.
6. **Interaktywny dashboard (Streamlit):**
   - Prezentacja wykresów świecowych oraz słupkowych dla wolumenu (z wykorzystaniem biblioteki Plotly).
   - Porównanie wartości rzeczywistych wolumenu z predykcjami modelu.
   - Interaktywne filtry: wybór analizowanej spółki, zmiana zakresu dat, wybór parametrów wygładzania i horyzontu predykcji.

---

## 3. Konfiguracja (`config.json`)

Kluczowe parametry projektu nie są wpisane na sztywno w kodzie źródłowym, lecz kontrolowane przez plik konfiguracyjny `config.json`, przykładowa początkowa struktura:

```json
{
  "tickers": ["AAPL", "MSFT", "NVDA", "AMZN"],
  "data_range": {
    "start_date": "2022-01-01",
    "end_date": "2024-01-01"
  },
  "database_path": "data/stocks.db",
  "prediction": {
    "target_column": "Volume",
    "forecast_horizons_days": [1, 5],
    "test_split_ratio": 0.2
  }
}
```

---

## 4. Struktura katalogów

```text
stock-volume-dynamics/
├── config.json               # Konfiguracja spółek, dat i parametrów modeli
├── requirements.txt          # Zależności biblioteczne
├── README.md                 # Dokumentacja projektu
├── PROJECT_SPECIFICATION.md  # Wymagania akademickie projektu
│
├── db/                     # Baza danych SQLite i ewentualne zrzuty cache
│   └── stocks.db
│
├── src/                      # Kod źródłowy projektu
│   ├── __init__.py
│   ├── data_loader.py        # Pobieranie danych z API
│   ├── database.py           # Inicjalizacja schematu i operacje SQL
│   ├── preprocessing.py      # Czyszczenie danych i inżynieria cech
│   └── model.py              # Trening, ewaluacja i inferencja modelu
│
└── app/                      # Interfejs użytkownika
    └── dashboard.py          # Aplikacja Streamlit
```

---

## 5. Wykorzystane technologie

- **Język:** Python 3.10+
- **Pobieranie i manipulacja danymi:** `requests` / `yfinance`, `pandas`, `numpy`
- **Baza danych:** `sqlite3` (moduł biblioteki standardowej Pythona)
- **Modelowanie i statystyka:** `scikit-learn`
- **Wizualizacja:** `plotly`
- **Aplikacja webowa / Dashboard:** `streamlit`

---

## 6. Uruchomienie projektu

### Krok 1. Klonowanie i przygotowanie środowiska wirtualnego
```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### Krok 2. Instalacja zależności
```bash
pip install -r requirements.txt
```

### Krok 3. Pobranie danych i zasilenie bazy
```bash
python src/data_loader.py
```

### Krok 4. Trenowanie modelu i generowanie prognoz
```bash
python src/model.py
```

### Krok 5. Uruchomienie dashboardu
```bash
streamlit run app/dashboard.py
```
Aplikacja zostanie uruchomiona w domyślnej przeglądarce pod adresem `http://localhost:8501`.

