# Część projektowa – Projekt analizy danych

## Informacje ogólne i organizacja

- **Forma zajęć:** spotkania 3-godzinne
- **Zespoły:** realizacja w zespołach 2–5 osobowych
- **Tematyka:** projekt powinien dotyczyć konkretnego, jasno określonego problemu

### Cel projektu

Celem projektu jest wykonanie spójnego rozwiązania przechodzącego przez pełny proces pracy z danymi:

```text
dane ➔ preprocessing ➔ baza danych ➔ analiza/model ➔ dashboard ➔ wnioski
```

---

## Wymagania i etapy realizacji projektu

### Etap 1. Wybór problemu i danych

Zespół wybiera:
- **Temat projektu**
- **Problem**, który chce przeanalizować
- **Źródło danych**
- **Podstawowe pytania**, na które projekt ma odpowiedzieć

**Dane mogą pochodzić np. z:**
- Publicznego API
- Plików CSV
- Portali Open Data
- Publicznych repozytoriów danych
- Innych legalnie dostępnych źródeł

---

### Etap 2. Pozyskanie i przygotowanie danych

**Zakres prac:**
- Pobranie danych
- Analiza struktury danych
- Sprawdzenie jakości danych
- Obsługa brakujących wartości
- Usunięcie lub analiza błędnych obserwacji
- Niezbędne transformacje
- Przygotowanie danych do dalszej analizy
- **Eksploracyjna analiza danych (EDA)**

**Można wykorzystać m.in.:**
- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Inne odpowiednie biblioteki

---

### Etap 3. Baza danych

Wybrany wariant danych należy zapisać w relacyjnej bazie danych.

**Należy:**
1. Zaprojektować strukturę tabel
2. Stworzyć bazę danych
3. Zapisać do niej dane
4. Wykonać odczyt danych

- **Rekomendowana baza:** `SQLite`
- *Możliwe jest wykorzystanie innej relacyjnej bazy danych.*

---

### Etap 4. Wykorzystanie danych z bazy

Samo zapisanie danych do bazy nie jest wystarczające. **Baza musi zostać wykorzystana w co najmniej jednym kolejnym etapie projektu**, np.:
- Odczyt danych do modelowania
- Zapis wyników modelowania i ich późniejszy odczyt
- Pobieranie danych przez dashboard
- Zapis wyników analiz
- Pobieranie danych do generowania wizualizacji

---

### Etap 5. Modelowanie lub pogłębiona analiza

Projekt powinien zawierać model matematyczny/statystyczny lub model uczenia maszynowego odpowiadający wybranemu problemowi.

**Przykładowe zagadnienia:**
- Regresja
- Klasyfikacja
- Prognozowanie
- Analiza szeregu czasowego
- Segmentacja
- Predykcja sprzedaży
- Predykcja cen
- Analiza zachowania użytkowników

**Model należy nie tylko stworzyć, ale również:**
- Odpowiednio przygotować dane
- Dobrać metodę
- Ocenić jej działanie
- Zinterpretować wyniki

#### Alternatywa: Pogłębiona analiza problemu
Jeżeli charakter danych nie uzasadnia budowania modelu predykcyjnego, możliwe jest wykonanie **pogłębionej analizy problemu** *(wariant ten powinien zostać wcześniej uzgodniony z prowadzącym)*.

Może ona obejmować m.in.:
- Analizę statystyczną
- Testowanie hipotez
- Analizę zależności
- Analizę trendów
- Porównanie grup
- Analizę czynników wpływających na dane zjawisko

---

### Etap 6. Dashboard

Ostatnim elementem projektu jest przygotowanie interaktywnego dashboardu.

**Można wykorzystać np.:**
- Dash
- Streamlit
- Power BI
- Inne uzgodnione narzędzie

**Dashboard powinien zawierać:**
- Najważniejsze informacje dotyczące analizowanego problemu
- Kluczowe wykresy
- Najważniejsze wyniki
- Wnioski z przeprowadzonej analizy

**Wymagane elementy interaktywne:**
*(Dashboard nie może być wyłącznie statycznym zestawem wykresów)*
- Wybór zakresu danych
- Wybór analizowanej zmiennej
- Wybór kategorii
- Zmiana okresu
- Filtrowanie danych
- Zmiana wybranych parametrów
- Prezentacja wyników dla wybranego przypadku
