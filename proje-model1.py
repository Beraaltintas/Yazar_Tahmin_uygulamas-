import joblib
from zemberek import TurkishSentenceNormalizer, TurkishSpellChecker, TurkishSentenceExtractor, TurkishMorphology
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Zemberek nesnesini oluştur
morphology = TurkishMorphology.create_with_defaults()
sc = TurkishSpellChecker(morphology)
extractor = TurkishSentenceExtractor()

# Veri kümesini oluştur
authors = ['asu-maro', 'atilla-gokce', 'levent-kalkan', 'tunca-bengin', 'zafer-sahin']
veri_kumesi = {}


for author in authors:
    veri_kumesi[author] = []

    for i in range(1, 20):  # 20 tane yazı mevcut
        path = f'/Users/bera/Documents/projeler/author-dataset kopyası/val/{author}/{author}{i}.txt'

        with open(path, 'r', encoding='utf-8') as file:
            metin = file.read()
            metin = TurkishSentenceNormalizer(morphology).normalize(metin)
            metin = ' '.join([sc.suggest_for_word(word)[0] if sc.suggest_for_word(word) else word for word in metin.split()])
            cümleler = extractor.extract_sentences(metin)  # Cümle ayıklama işlemi
            veri_kumesi[author].extend(cümleler)

# Veriyi eğitim ve test setlerine böl
# Tüm metinleri tek bir listede topla
all_texts = []
all_labels = []

for author, metinler in veri_kumesi.items():
    all_texts.extend(metinler)
    all_labels.extend([author] * len(metinler))

train_texts, test_texts, train_labels, test_labels = train_test_split(all_texts, all_labels, test_size=0.33, random_state=42)


# Modeli eğit
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(train_texts, train_labels)

# Test setinde modeli değerlendir
predictions = model.predict(test_texts)
accuracy = accuracy_score(test_labels, predictions)
print(f"Test Doğruluğu: {accuracy}")

# Modeli kaydet
model_dosya_adi = "tahmin_modeli12.joblib"
joblib.dump(model, model_dosya_adi)