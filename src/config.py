import os

try:
    DATASET_NAME = "csebuetnlp/xlsum"
    LANGUAGES = ["english", "turkish"]
    MODEL_NAME = "google/mt5-small"

    ROOT_DIR = "."
    DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")

    os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True)

except Exception as e:
    print(f"Hata: Konfigürasyon dosyası yüklenirken bir sorun oluştu. Hata: {e}")
