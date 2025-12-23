import sys
from src.data.downloader import download_data
from src.data.preprocessor import get_tokenizer, preprocess_function, validate_batch
from src.data.saver import save_to_disk
from src.config import DATASET_NAME, LANGUAGES, DATA_DIR

def process_and_validate_split(raw_dataset, tokenizer, split_name):
    """
    Tek bir split'i işle ve doğrula
    """
    print(f"\n {split_name.upper()} split işleniyor...")
    
    # Preprocessing uygula
    tokenized_dataset = raw_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer), 
        batched=True,
        remove_columns=raw_dataset.column_names,  # Orijinal sütunları temizle
        desc=f"Tokenizing {split_name}"
    )
    
    # Boş örnekleri filtrele
    tokenized_dataset = tokenized_dataset.filter(
        lambda example: len(example['input_ids']) > 0 and len(example['labels']) > 0
    )
    
    print(f" {split_name} işlendi: {len(tokenized_dataset)} örnek")
    
    # İlk birkaç örneği kontrol et
    if len(tokenized_dataset) > 0:
        sample = tokenized_dataset[0]
        print(f" Örnek shape kontrolü:")
        print(f"   - input_ids: {len(sample['input_ids'])}")
        print(f"   - attention_mask: {len(sample['attention_mask'])}")  
        print(f"   - labels: {len(sample['labels'])}")
        
        # Decode kontrolü
        decoded_input = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        decoded_labels = tokenizer.decode([id for id in sample['labels'] if id != -100], skip_special_tokens=True)
        print(f"   - Input preview: {decoded_input[:100]}...")
        print(f"   - Label preview: {decoded_labels[:100]}...")
    
    return tokenized_dataset

if __name__ == "__main__":
    try:
        print(" Veri hazırlama süreci başlatılıyor...")
        
        # 1. Veri setini indir
        print(" Veri indiriliyor...")
        raw_datasets = download_data(DATASET_NAME, LANGUAGES)
        if raw_datasets is None:
            print(" Veri indirme işlemi başarısız. Program sonlandırılıyor.")
            sys.exit(1)
        
        print(f" Ham veri indirildi:")
        for split, dataset in raw_datasets.items():
            print(f"   - {split}: {len(dataset)} örnek")
        
        # 2. Tokenizer yükle
        print("\n Tokenizer yükleniyor...")
        tokenizer = get_tokenizer()
        if tokenizer is None:
            print(" Tokenizer yükleme işlemi başarısız. Program sonlandırılıyor.")
            sys.exit(1)
        
        print(f" Tokenizer yüklendi: {tokenizer.name_or_path}")
        print(f"   - Vocab size: {tokenizer.vocab_size}")
        print(f"   - Pad token: {tokenizer.pad_token}")
        
        # 3. Her split'i işle
        tokenized_datasets = {}
        for split_name, raw_dataset in raw_datasets.items():
            tokenized_datasets[split_name] = process_and_validate_split(
                raw_dataset, tokenizer, split_name
            )
        
        # 4. Final kontrol
        print(f"\n İşlenmiş veri özeti:")
        total_samples = 0
        for split, dataset in tokenized_datasets.items():
            sample_count = len(dataset)
            total_samples += sample_count
            print(f"   - {split}: {sample_count} örnek")
        
        print(f"   - TOPLAM: {total_samples} örnek")
        
        # 5. Veriyi kaydet
        if total_samples > 0:
            print(f"\n Veri kaydediliyor: {DATA_DIR}")
            save_to_disk(tokenized_datasets, DATA_DIR)
            print(" Veri hazırlama tamamlandı!")
        else:
            print(" İşlenmiş veri bulunamadı, kaydetme atlandı.")
            
    except Exception as e:
        print(f" Ana program akışında beklenmedik bir hata oluştu:")
        print(f"   Hata: {e}")
        import traceback
        traceback.print_exc()
