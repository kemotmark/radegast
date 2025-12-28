import requests
from bs4 import BeautifulSoup

# #############################################################################
# USTAWIENIA
# #############################################################################
# Zastąp poniższy URL adresem strony, z której chcesz pobrać dane
URL = "https://twoj-adres-strony.pl"  # <-- ZMIEŃ NA RZECZYWISTY URL

# Jeśli strona wymaga nagłówków (np. user-agent), możesz je dodać
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# #############################################################################
# FUNKCJA GŁÓWNA
# #############################################################################
def pobierz_ceny_z_strony(url):
    try:
        # Pobranie zawartości strony
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()  # Sprawdza czy nie ma błędów HTTP (np. 404)
    except requests.exceptions.RequestException as e:
        print(f"BŁĄD POŁĄCZENIA: {e}")
        return []

    # Parsowanie HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # #########################################################################
    # KROK 1: TWORZENIE MAPOWANIA ID -> OPIS
    # #########################################################################
    mapping = {}

    # A. Mapowanie dla SELECT (waga)
    select_waga = soup.find('select', {'name': 'select_usluga1wybor1'})
    if select_waga:
        for option in select_waga.find_all('option'):
            value = option.get('value')
            if value:
                mapping[value] = option.text.strip()

    # B. Mapowanie dla RADIO (rodzaj przesyłki i format)
    # Szukamy WSZYSTKICH inputów typu radio oraz odpowiadających im labeli
    for input_tag in soup.find_all('input', type='radio'):
        input_id = input_tag.get('id')
        input_value = input_tag.get('value')
        
        if not input_id or not input_value:
            continue

        # Znajdź label, który ma atrybut `for` pasujący do ID inputa
        label = soup.find('label', attrs={'for': input_id})
        if label:
            # Pobierz czysty tekst (bez nadmiarowych spacji i tagów)
            opis = label.get_text(separator=" ", strip=True)
            mapping[input_value] = opis

    # #########################################################################
    # KROK 2: POBIERANIE CEN Z SEKcji .ceny
    # #########################################################################
    ceny_list = []
    cena_spans = soup.select('.ceny span.cena')

    if not cena_spans:
        print("Nie znaleziono elementów z klasą '.cena' w sekcji .ceny.")
        return []

    for span in cena_spans:
        klasy = span.get('class', [])
        # Pozostawiamy tylko klasy zaczynające się od 'wybor_'
        wybory = [klasa for klasa in klasy if klasa.startswith('wybor_')]

        # Jeśli nie ma wymaganych klas, pomijamy
        if len(wybory) < 3:  # oczekujemy 3 parametrów: waga, rodzaj, format
            continue

        # Zamień ID na czytelny opis (lub zostaw ID, jeśli nie ma w mapowaniu)
        opisy = [mapping.get(wyb, wyb) for wyb in wybory]
        cena = span.get_text(strip=True)

        ceny_list.append({
            "kombinacja": " | ".join(opisy),
            "cena": f"{cena} PLN"
        })

    return ceny_list

# #############################################################################
# URUCHOMIENIE SKRYPTU
# #############################################################################
if __name__ == "__main__":
    wyniki = pobierz_ceny_z_strony(URL)

    if wyniki:
        print(f"{'Dostępne kombinacje cenowe':<60} | {'Cena':<10}")
        print("-" * 75)
        for w in wyniki:
            print(f"{w['kombinacja']:<60} | {w['cena']:<10}")
    else:
        print("Nie udało się pobrać cen.")
