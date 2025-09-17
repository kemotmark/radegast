import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import time
import csv
import os

# --- Przykładowe dane PKD ---
pkd_data = [
   # {
   #     "kod": "PKD_01.11.Z",
   #     "nazwa": "Uprawa zbóż, roślin strączkowych i oleistych",
    #    "opis": """Uprawa zbóż (bez ryżu), roślin strączkowych i oleistych na uprawę zbóż, takich jak:
#pszenica, kukurydza, proso, sorgo, jęczmień, żyto, owies,
#pozostałe zboża, gdzie indziej niesklasyfikowane,
#uprawę roślin strączkowych, takich jak: fasola, bób, ciecierzyca, wspięga chińska, soczewica, łubin, groch,
#oraz roślin oleistych: soja, orzeszki ziemne, bawełna, rącznik, siemię lniane, gorczyca, rzepak, szafran, sezam, słonecznik itp."""
 #   },
 #   {
 # "kod":"PKD_07.10.Z",     
 # "nazwa":"Górnictwo rud żelaza",
#"opis":"""Wydobywanie i przeróbka rud żelaza.

#Podklasa ta obejmuje:
#górnictwo rud wartościowych, głównie ze względu na zawartość żelaza,
#wzbogacanie rud żelaza.

#Podklasa ta nie obejmuje:
#wydobywania i przeróbki pirytów i magnetopirytów, z wyłączeniem prażenia pirytów, sklasyfikowanego w 08.91.Z,"""
 #   },
]


df = pd.DataFrame(pkd_data)

# --- Model embeddingów ---
print("🔄 Ładowanie modelu embeddingów...")
model = SentenceTransformer("intfloat/multilingual-e5-base")

# --- Generowanie embeddingów ---
print("🔄 Generowanie embeddingów PKD...")
texts = ('query:'+df['kod'] + "  passage:" + df['opis'] + "-" + df['nazwa']).tolist()
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# --- Budowanie indeksu FAISS ---
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# --- Lokalny endpoint LLM ---
_API_URL = "http://localhost:1234/v1/chat/completions"

def call_llm(system_prompt, user_prompt):
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.8,
        "max_tokens": 500,
        "stream": False,
        "model": "bielik-7b-instruct-v0.1"  # dopasuj nazwę modelu do lokalnego serwera
    }
    response = requests.post(_API_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# --- Generowanie wizytówek dla jednego PKD ---
def generate_multiple_business_cards(pkd_code, count=5, top_k=1):
    # Embedding zapytania
    q_emb = model.encode([pkd_code], convert_to_numpy=True)
    distances, indices = index.search(q_emb, top_k)

    # Pobierz kontekst z najbardziej podobnych PKD
    docs = []
    for idx in indices[0]:
        kod = df.iloc[idx]['kod']
        nazwa = df.iloc[idx]['nazwa']
        opis = df.iloc[idx]['opis']
        docs.append(f"PKD {kod} - {nazwa}:\n{opis}")

    context = "\n\n".join(docs)

    system_prompt = (
        "Jesteś ekspertem od tworzenia profesjonalnych, krótkich i marketingowych wizytówek firm. "
        "Na podstawie informacji kontekst działalności."
    )

    results = []

    for i in range(count):
        user_prompt = (
            f"Oto kontekst działalności:\n\n{context}\n\n"
            "Na jego podstawie wygeneruj  profesjonalny opis firmy"
            "W konekscie znajdują się również linie 'Podklasa ta nie obejmuje:' dla nich również dopisz właściwy kod PKD i wygeneruj wizytówkę. "
            "Każdy opis w osobnej linii, bez numeracji, bez markdown. TYLKO  tekst w języku polskim"
        )

        try:
            answer = call_llm(system_prompt, user_prompt)
            results.append(answer)
            print(f"✅ {pkd_code} - wygenerowano zestaw {i+1}")
        except Exception as e:
            print(f"❌ Błąd przy generowaniu dla {pkd_code}, prób {i+1}: {e}")
            results.append("")
        time.sleep(1)  # Ograniczenie liczby zapytań
    return results


# --- Główna funkcja ---
def main_generate(pkd_list, output_csv="wizytowki_firm.csv", wizytowki_na_kod=5):
    results = []
    total = len(pkd_list) * wizytowki_na_kod * 20  # 20 opisów na jeden zestaw
    counter = 0

    for pkd_code in pkd_list:
        wizytowki_zestawy = generate_multiple_business_cards(pkd_code, count=wizytowki_na_kod)

        for blok in wizytowki_zestawy:
            lines = blok.strip().split("\n")
            for line in lines:
                opis = line.strip().strip('"')
                if opis:
                    counter += 1
                    results.append({"pkd": pkd_code, "opis": opis})
                    print(f"Postęp: {counter}/{total}")

    # Zapis do CSV
    write_header = not os.path.exists(output_csv)

    with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["pkd", "opis"])
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    print(f"\n📁 Zapisano {len(results)} wizytówek do pliku: {output_csv}")


# --- Uruchomienie ---
if __name__ == "__main__":
    unikatowe_kody = list(set(item["kod"] for item in pkd_data))
    pkd_list_to_generate = unikatowe_kody #[
      #  "11.07.Z",
       
    #]

    main_generate(pkd_list_to_generate, wizytowki_na_kod=5)
