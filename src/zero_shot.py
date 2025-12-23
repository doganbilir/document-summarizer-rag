import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import torch



PROCESSED_DATA_PATH = "/content/drive/MyDrive/document-summarizer-rag/data/processed"
MODEL_NAME = "google/mt5-small"

def get_zero_shot_summary(text, tokenizer, model):
    """
    Önceden eğitilmiş modeli kullanarak metin özetler.
    """
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        input_ids,
        max_length=150,
        min_length=40,
        num_beams=4,
        early_stopping=True
    )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def evaluate_zero_shot_model():
    """
    Önceden eğitilmiş modelin performansını test verisi üzerinde değerlendirir.
    """
    try:
        # Burada 'test' alt klasörünü belirtiyoruz
        test_dataset = load_from_disk(os.path.join(PROCESSED_DATA_PATH, 'test'))
    except Exception as e:
        print(f"Hata: Veri seti '{PROCESSED_DATA_PATH}/test' yolunda bulunamadı. Lütfen yolu kontrol edin.")
        raise e

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    test_subset = test_dataset.select(range(100))
    
    for example in test_subset:
        original_text = example['text']
        true_summary = example['summary']
        zero_shot_summary = get_zero_shot_summary(original_text, tokenizer, model)
        
        print("-" * 50)
        print(f"Orijinal Metin:\n{original_text}\n")
        print(f"Gerçek Özet:\n{true_summary}\n")
        print(f"Zero-Shot Özet:\n{zero_shot_summary}\n")
        print("-" * 50)

if __name__ == "__main__":
    evaluate_zero_shot_model()
