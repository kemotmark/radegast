import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === PARAMETRY ===
CSV_KLASA= "pkd.csv"              # Plik z kodami PKD
EMBEDDINGS_PATH = "data/" 
CSV_PATH = "data/" # Gdzie zapisać embeddingi
#MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # Model do embeddingów (obsługuje polski)
MODEL_NAME = "intfloat/multilingual-e5-large"
#MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

def create_embeddings(df,klasa,nazwa_modelu=MODEL_NAME):
    # === KROK 1: Wczytanie danych ===
    print("📄 Wczytywanie danych PKD...")

    # Sprawdź czy wymagane kolumny istnieją
    required_columns = {"PKD", "SEKCJA", "Opis"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Plik {klasa} musi zawierać kolumny: {required_columns}")

    # === KROK 2: Przygotowanie tekstów ===
    print("🧾 Przygotowywanie tekstów do embeddingów...")
    texts = (df["PKD"].fillna("") + " - " + df["Opis"].fillna("")).tolist()

    # === KROK 3: Wczytanie modelu ===
    print(f"🔍 Ładowanie modelu: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # === KROK 4: Generowanie embeddingów ===
    print("⚙️ Generowanie embeddingów...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # === KROK 5: Zapis do pliku ===
    print(f"💾 {klasa} - Zapisywanie do katalogu: {EMBEDDINGS_PATH}")
    np.save(EMBEDDINGS_PATH+klasa+'.npy', embeddings)

    print("✅ Gotowe! Embeddingi zapisane.")


def Predykcja(klasa='D',txt="Nasza firma zajmuje się sprzedarzą energi elektrycznej, wytwarzaniem i dystrybucją prądu."):
    embeddings = np.load(f"{EMBEDDINGS_PATH}/{klasa}.npy")

    model = SentenceTransformer(MODEL_NAME)
    # Przygotuj zapytanie
    query_vec = model.encode([txt])

    # Oblicz podobieństwo
    similarities = cosine_similarity(query_vec, embeddings)[0]

    # Posortuj
    top_indices = similarities.argsort()[-5:][::-1]
    df=pd.read_csv(f"{CSV_PATH}/{klasa}.csv", dtype=str)    
    lst=[]
    # Wyświetl top 5 najbardziej podobnych kodów
    for i in top_indices:
        print(f"{df.iloc[i]['PKD']} – {df.iloc[i]['Opis']}")
        lst.append(f"{df.iloc[i]['PKD']} – {df.iloc[i]['Opis']}")
        #print(f"{df.iloc[i]['Opis']}")
        print(f"Similarity: {similarities[i]:.4f}\n")
    return max(similarities), lst

df=pd.read_csv("StrukturaPKD2025.csv", sep=";", dtype=str, header=1)

df.columns=['SEKCJA', 'Grupa', 'Klasa', 'PKD', 'Opis']
print(df.head())

y = df[df['Klasa'].astype(str).str.len() == 7]['Klasa'].unique()
print(set(y))

#sekcja_indices = df[df['SEKCJA'].str.contains('SEKCJA', na=False)].index
def find_section(df, sek='B', sek_next='C'):
    # Znajdź początek i koniec sekcji B
    sekcja='SEKCJA '+sek
    sekcja_next='SEKCJA '+sek_next

    start_idx = df[df['SEKCJA'].str.contains(sekcja, na=False)].index[0]
    end_idx = df[df['SEKCJA'].str.contains(sekcja_next, na=False)].index[0]

    print(start_idx)# end_idx)
    # Wytnij wiersze należące do sekcji B
    sekcja_df = df.iloc[start_idx:end_idx]
    print(sekcja_df)
  
    sekcja_a_value = df.loc[df['SEKCJA'] == sekcja, 'Klasa'].values[0]
    print(sekcja_a_value)

    sekcja_df['tytul'] = sekcja_a_value

    sekcja_df['SEKCJA']=sekcja

    filtered_df = sekcja_df[sekcja_df['PKD'].astype(str).str.len() == 7]

    filtered_df=filtered_df[['SEKCJA','PKD','tytul','Opis']]
    filtered_df.to_csv(f'{CSV_PATH}{sek}.csv', index=False)    

    return filtered_df[['SEKCJA','PKD','tytul','Opis']]

#for sek in ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']: #,'O','P','Q','R','S','T','U']:
#    x=find_section(df, sek, chr(ord(sek)+1))    
#    create_embeddings(x, sek)

wynik ={}
odp={}
for sek in ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']: #,'O','P','Q','R','S','T','U']:
    print(f"--- Sekcja {sek} ---")
    wynik[sek], opis = Predykcja(klasa=sek,txt="""  SKŁAD WĘGLA
Serdecznie zapraszamy do naszego  sklepu sprzedażowego
kostkę,

groszek,

orzech,

ekogroszek,

ekomiał,

pellet.

Dokładna waga, szybka dostawa

                     
""")
    odp[sek]=opis[0]
    print(f"Opis: {opis[0]}")

print(wynik)
najwiekszy_klucz = max(wynik, key=wynik.get)

print(f"Największy klucz: {najwiekszy_klucz} o wartości {wynik[najwiekszy_klucz]}")
print(f"Opis: {odp[najwiekszy_klucz]}")
