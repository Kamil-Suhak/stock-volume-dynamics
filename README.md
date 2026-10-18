# Stock Volume Dynamics – Analiza i prognozowanie wolumenu obrotu giełdowego

Projekt akademicki realizowany w ramach przedmiotu z zakresu analizy danych. Aplikacja realizuje pełny potok przetwarzania danych (ang. *end-to-end data pipeline*): od pobrania historycznych godzinowych notowań giełdowych przez API, przez czyszczenie i składowanie w relacyjnej bazie danych SQLite, po inżynierię cech, modelowanie statystyczne i uczenie maszynowe oraz prezentację wyników w interaktywnym dashboardzie Streamlit.

---

## 1. Cel i zakres projektu

Głównym celem projektu jest analiza dynamiki wolumenu obrotu akcjami (ang. *trading volume*) dla wybranej grupy spółek (m.in. AAPL, MSFT, NVDA, TSLA) oraz budowa modelu prognozującego wolumen w horyzoncie godzinowym.

Wolumen obrotu jest kluczową zmienną rynkową niosącą informacje o płynności waloru i aktywności inwestorów. Projekt odpowiada na następujące pytania badawcze:
- Jak kształtuje się rozkład oraz dzienna i tygodniowa sezonowość wolumenu w godzinach sesji regularnej (RTH)?
- W jakim stopniu przeszłe notowania, wskaźniki zmienności oraz rozpiętość cenowa determinują przyszły wolumen?
- Z jaką dokładnością modele regresyjne (Ridge, HistGradientBoosting) prognozują odchylenia wolumenu od typowej krzywej dziennej?

---

## 2. Architektura systemu i etapy realizacji

Zgodnie z wymaganiami projektowymi ([PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md)), potok składa się z sześciu etapów:

```text
[Polygon / Massive API]
       │
       ▼
1. Pobranie danych godzinowych i buforowanie (`src/ingest.py`)
       │
       ▼
2. Czyszczenie i analiza wstępna (`src/eda.py`, `src/features.py`)
       │
       ▼
3. Relacyjna baza danych SQLite (`data/market.db`, `src/db.py`)
       │
       ├──► 4. Ekstrakcja danych do inżynierii cech
       │           │
       │           ▼
       │      5. Trening i walidacja modeli (`src/train.py`)
       │           │
       │           ▼
       └──► 6. Zasilenie interaktywnego dashboardu (`app.py`)
```

### Opis modułów:
1. **Pozyskanie danych (API):**
   - Pobieranie danych godzinowych (OHLCV + liczba transakcji) oraz danych fundamentalnych za pośrednictwem API Polygon / Massive.
   - Buforowanie odpowiedzi JSON w katalogu `raw_data/` w celu optymalizacji zapytań i limitów API.
2. **Czyszczenie i preprocessing:**
   - Standaryzacja strefy czasowej (UTC na czas nowojorski ET) i filtracja sesji regularnej (RTH: 09:30–16:00).
   - Transformacja logarytmiczna wolumenu ($\ln(\text{Volume})$) stabilizująca wariancję.
   - Wstępna eksploracja statystyczna (skrypt `src/eda.py`).
3. **Baza danych (SQLite):**
   - Baza składowana lokalnie w `data/market.db`.
   - Tabele relacyjne: `symbols` (metadane spółek), `hourly_bars` (notowania godzinowe), `predictions` (wyniki modeli), `model_benchmarks` (metryki walidacji).
4. **Wykorzystanie bazy w potoku:**
   - Moduł modelowania zasilany jest bezpośrednio zapytaniami SQL z bazy.
   - Predykcje modeli i metryki ewaluacyjne są zapisywane z powrotem do bazy, skąd pobiera je aplikacja Streamlit.
5. **Modelowanie predykcyjne:**
   - Inżynieria cech: 15-dniowy kroczący profil mediany godzinowej (baseline RVOL), opóźnienia czasowe (lags), rozpiętość świecy (High - Low), stopy zwrotu i zmienność.
   - Dwa tryby modeli: *Forward* (wyłącznie dane historyczne) oraz *Explanatory* (z uwzględnieniem współbieżnej dynamiki cenowej).
   - Algorytmy: regresja grzbietowa (*Ridge*) oraz gradient boosting (*HistGradientBoostingRegressor*).
   - Ewaluacja z użyciem 5-krotnej kroczącej walidacji czasowej (*TimeSeriesSplit*) z metrykami: MAE, RMSE, MAPE.
6. **Interaktywny dashboard (`app.py`):**
   - Wykresy świecowe i wolumenowe (Plotly) z przełączaniem skali (dzienna / godzinowa).
   - Analiza sezonowości: profil krzywej U z pasmem percentyli (25–75%) oraz heatmapa dzień-godzina.
   - Zestawienie prognoz z wartościami rzeczywistymi i kalkulator wpływu wielkości zlecenia na rynek (*Participation Rate*).

---

## 3. Struktura katalogów

```text
stock-volume-dynamics/
├── app.py                    # Interaktywny dashboard Streamlit
├── requirements.txt          # Zależności biblioteczne
├── README.md                 # Dokumentacja projektu
├── PROJECT_SPECIFICATION.md  # Wymagania akademickie projektu
│
├── data/                     # Relacyjna baza danych SQLite
│   └── market.db
│
├── raw_data/                 # Zbuforowane odpowiedzi API (JSON per spółka)
│   └── <TICKER>/
│
└── src/                      # Kod źródłowy potoku danych i modeli
    ├── __init__.py
    ├── db.py                 # Inicjalizacja schematu bazy i operacje SQL
    ├── ingest.py             # Pobieranie danych z API i zasilanie bazy
    ├── eda.py                # Eksploracyjna analiza danych
    ├── features.py           # Inżynieria cech i obliczanie baseline'u
    ├── train.py              # Trening modeli, walidacja TimeSeriesSplit
    └── main.py               # Główny skrypt orkiestrujący zasilanie i trening
```

---

## 4. Wykorzystane technologie

- **Język:** Python 3.10+
- **Pobieranie danych:** `requests`, Massive / Polygon.io API
- **Baza danych:** `sqlite3` (wbudowany moduł Pythona)
- **Manipulacja danymi i obliczenia:** `pandas`, `numpy`
- **Uczenie maszynowe:** `scikit-learn` (Ridge, HistGradientBoosting, TimeSeriesSplit)
- **Wizualizacja:** `plotly`
- **Dashboard:** `streamlit`

---

## 5. Uruchomienie projektu

### Krok 1. Przygotowanie środowiska wirtualnego
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

### Krok 3. Pobranie danych i trening modeli
Uruchomienie głównego skryptu pobiera dane dla skonfigurowanych spółek, aktualizuje bazę SQLite i trenuje modele predykcyjne:
```bash
python src/main.py
```
*(Opcjonalnie)* Uruchomienie skryptu wstępnej analizy eksploracyjnej:
```bash
python src/eda.py
```

### Krok 4. Uruchomienie dashboardu
```bash
streamlit run app.py
```
Aplikacja zostanie uruchomiona lokalnie w przeglądarce pod adresem `http://localhost:8501`.
