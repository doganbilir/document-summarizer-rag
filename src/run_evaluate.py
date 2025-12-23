# src/evaluate.py
import os
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import json

# Lütfen bu yolu kendi Drive'ınızdaki doğru yolla güncelleyin.
PROCESSED_DATA_PATH = "/content/drive/MyDrive/document-summarizer-rag/data/processed"
MODEL_NAME = "google/mt5-small"

def calculate_rouge_scores():
    """
    Modelin zero-shot performansını ROUGE metrikleriyle değerlendirir.
    """
    try:
        test_dataset = load_from_disk(os.path.join(PROCESSED_DATA_PATH, 'test'))
    except Exception as e:
        print(f"Hata: Veri seti '{PROCESSED_DATA_PATH}/test' yolunda bulunamadı. Lütfen yolu kontrol edin.")
        return None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # 100 örnekle hızlı bir test yapalım.
    test_subset = test_dataset.select(range(100))
    
    all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for example in test_subset:
        original_text = example['text']
        true_summary = example['summary']
        
        input_ids = tokenizer.encode("summarize: " + original_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            input_ids,
            max_length=150,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )
        zero_shot_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        scores = scorer.score(true_summary, zero_shot_summary)
        
        all_scores['rouge1'].append(scores['rouge1'].fmeasure)
        all_scores['rouge2'].append(scores['rouge2'].fmeasure)
        all_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    avg_scores = {key: sum(val) / len(val) for key, val in all_scores.items()}
    
    # Sonuçları reports klasörüne kaydet
    report_path = "reports/zero_shot_rouge_scores.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(avg_scores, f, indent=4)
        
    print("Zero-Shot ROUGE Skorları:")
    print(json.dumps(avg_scores, indent=4))
    print(f"\nSkorlar '{report_path}' dosyasına kaydedildi.")

if __name__ == "__main__":
    calculate_rouge_scores()