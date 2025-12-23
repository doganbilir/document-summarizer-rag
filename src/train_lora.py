import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    GenerationConfig,
    TrainerCallback,
)
from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from evaluate import load
import json

os.environ["WANDB_DISABLED"] = "true"
hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()
set_seed(42)

PROJECT_ROOT = "/content/drive/MyDrive/document-summarizer-rag-2"
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "lora_adapter_v2_fast_eval")
LOGGING_DIR = os.path.join(PROJECT_ROOT, "logs", "lora_adapter_v2_fast_eval")
MODEL_NAME = "google/mt5-small"
rouge = load("rouge")
tokenizer = None

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    
    print("\n--- Örnek Model Çıktıları ---")
    for i in range(min(3, len(decoded_preds))):
        print(f"Tahmin {i+1}: '{decoded_preds[i]}'")
        print(f"Gerçek {i+1}: '{decoded_labels[i]}'")
        print("-" * 10)
    print("---------------------------------")
    
    result = rouge.compute(
        predictions=decoded_preds, references=decoded_labels,
        use_stemmer=True, rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"]
    )
    return result

def get_lora_model(model):
    model.train()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q", "k", "v", "o", "wi", "wo"],
        lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM",
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model

class LossPrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        if "loss" in logs and "learning_rate" in logs:
            print(f"[Eğitim] Adım={state.global_step} | Loss={logs['loss']:.4f} | Öğrenme Oranı={logs['learning_rate']:.2e}")
        if "eval_loss" in logs:
            r1 = logs.get("eval_rouge1", 0.0); rL = logs.get("eval_rougeL", 0.0)
            print(f"[Değerlendirme] Adım={state.global_step} | Eval Loss={logs['eval_loss']:.4f} | ROUGE-1={r1:.4f} | ROUGE-L={rL:.4f}")

def train_lora_model():
    global tokenizer
    
    try:
        train_dataset = load_from_disk(os.path.join(PROCESSED_DATA_PATH, "train"))
        eval_dataset  = load_from_disk(os.path.join(PROCESSED_DATA_PATH, "validation"))
        
        
        print("Veri seti 30,000 eğitim ve 3,000 doğrulama.")
        train_dataset = train_dataset.select(range(30000))
        eval_dataset = eval_dataset.select(range(3000))
        
        print(f" Küçültülmüş Veri seti boyutları - Eğitim: {len(train_dataset)}, Doğrulama: {len(eval_dataset)}")
        if len(train_dataset) == 0: raise ValueError("Eğitim veri seti boş!")
    except Exception as e:
        print(f" Veri yükleme hatası: {e}")
        return

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    USE_BF16 = False 

    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float32,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    lora_model = get_lora_model(model)

    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.max_new_tokens = 160
    generation_config.num_beams = 5
    generation_config.no_repeat_ngram_size = 3
    generation_config.length_penalty = 1.5
    generation_config.early_stopping = True
    lora_model.generation_config = generation_config

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=lora_model,
        label_pad_token_id=-100, pad_to_multiple_of=8
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_dir=LOGGING_DIR,
        logging_strategy="steps",
        logging_steps=50, # Logları biraz seyrekleştirelim
        
        # --> GEREKLİ GÜNCELLEME: Kütüphane uyumluluğu için parametre ismi düzeltildi.
        eval_strategy="steps", 
        save_strategy="steps",
        
        # --> İSTEĞİNİZ ÜZERE: ROUGE skorlarını daha sık görmek için adımlar düşürüldü.
        eval_steps=500,
        save_steps=500,
        
        load_best_model_at_end=True,
        metric_for_best_model="eval_rougeL",
        greater_is_better=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=False,
        bf16=USE_BF16,
        gradient_checkpointing=True,
        predict_with_generate=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=lora_model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        tokenizer=tokenizer, data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[ EarlyStoppingCallback(early_stopping_patience=5), LossPrinterCallback() ],
    )

    print(" İyileştirilmiş yapılandırma ile eğitim başlıyor...")
    eff_bsz = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    print(f" Efektif Batch Boyutu: {eff_bsz}")
    print(f" Öğrenme Oranı: {training_args.learning_rate}")
    print(f" Epoch Sayısı: {training_args.num_train_epochs}")
    print(f" BF16 Desteği: {'Aktif' if USE_BF16 else 'Devre Dışı'}")

    lora_model.config.use_cache = False
    trainer.train()

    print(" Eğitim tamamlandı!")
    best_model_path = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_path)
    print(f" En iyi model şuraya kaydedildi: {best_model_path}")

    with open(os.path.join(OUTPUT_DIR, "training_results.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    return trainer

if __name__ == "__main__":
    train_lora_model()
