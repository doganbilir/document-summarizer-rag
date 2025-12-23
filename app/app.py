import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#  1. Define model and tokenizer paths 
model_name = "doganbilir/mt5-Turkish-English-Summarizer" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  2. Load model and tokenizer from the Hub 
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None

# 3. Define the summarization function 
def summarize(text, language_choice, min_length):
    if model is None:
        return "Model could not be loaded. Please check the Space logs."
    if not text or not text.strip():
        return "Please enter text to summarize."
    
    
    # Set the prompt based on the language choice
    if language_choice == "Turkish Text -> Turkish Summary":
        prompt = f"summarize: {text}"
    elif language_choice == "English Text -> English Summary":
        prompt = f"summarize: {text}"
        
    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
        
        summary_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=150,
            num_beams=5,
            min_new_tokens=int(min_length), # Use the value from the slider
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return summary.strip()
    except Exception as e:
        return f"An error occurred during summarization: {str(e)}"

# 4. Create the Gradio Interface 
iface = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(lines=15, placeholder="Paste the text you want to summarize here...", label="Text Input"),
        gr.Radio(
            
            ["Turkish Text -> Turkish Summary", "English Text -> English Summary"],
            label="Summarization Type",
            value="Turkish Text -> Turkish Summary"
        ),
        gr.Slider(
            minimum=10, 
            maximum=100, 
            value=10, # Default value for natural, short summaries
            step=5, 
            label="Minimum Summary Length (Tokens)",
            info="Increase this value to force the model to generate longer summaries."
        )
    ],
    outputs=gr.Textbox(lines=5, label="Generated Summary"),
    title="Multilingual Text Summarization Model",
    description="""This generates headline-style summaries for Turkish or English texts.
    The model was fine-tuned on 30,000 Turkish-English summary pairs using the LoRA technique, 
    based on the `google/mt5-small` model.""",
    
   
    examples=[
        ["Türkiye’de şehir parkları son yıllarda kapsamlı bir şekilde yenileniyor ve modern hâle getiriliyor. Yeni yürüyüş yolları, çocuk oyun alanları, spor alanları, bisiklet yolları ve dinlenme bölgeleri ekleniyor. Ziyaretçiler, temiz, güvenli ve konforlu bir ortamda vakit geçirebiliyor. Parklarda düzenlenen kültürel etkinlikler ve spor aktiviteleri vatandaşların sosyal yaşamını da destekliyor. Belediye yetkilileri, bu projelerle insanların daha sağlıklı, aktif ve sosyal bir yaşam sürmesini, doğayla bağlarını güçlendirmesini ve şehirdeki yaşam kalitesinin artmasını hedefliyor.", "Turkish Text -> Turkish Summary", 10],
        ["Electric cars are becoming increasingly popular worldwide. People choose electric vehicles because they produce less pollution and reduce fuel costs. Governments are promoting them by offering incentives and building more charging stations. Experts believe that electric cars could eventually replace most gasoline-powered vehicles, significantly decreasing greenhouse gas emissions and contributing to a cleaner environment.", "English Text -> English Summary", 10]
    ],
    theme="gradio/soft",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()