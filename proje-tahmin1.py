import joblib

model = "metin_siniflandirma_modeli1.joblib"
yuklenen_model = joblib.load(model)

authors = ['asu-maro', 'atilla-gokce', 'levent-kalkan', 'tunca-bengin', 'zafer-sahin']
veri_kumesi = {}

for author in authors:
    veri_kumesi[author] = []

    for i in range(1, 2):
        tahmin_metni = f'/Users/bera/Documents/projeler/author-dataset kopyası/yeni_yazilar/{author}.txt'

        with open(tahmin_metni, 'r', encoding='utf-8') as file:
            metin = file.read()
            tahmin = yuklenen_model.predict([metin])
        print(f"Tahmin edilecek yazının sahibi: ({author}): Tahmin: {tahmin[0]}")