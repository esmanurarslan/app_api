# app.py
import os
import io
from flask import Flask, request, jsonify
from PIL import Image
import torch # torch import'u en başta olmalı
from transformers import AutoModelForCausalLM, AutoProcessor

# ------------------------------------
# 1. Uygulama ve Model Ayarları
# ------------------------------------
app = Flask(__name__)

# PythonAnywhere'de GPU olmayacağı için sabit değerler:
DEVICE = "cpu"
TORCH_DTYPE = torch.float32
MODEL_NAME = "microsoft/Florence-2-base" # Jupyter'deki gibi

print(f"API Başlatılıyor... Kullanılacak Model: {MODEL_NAME}, Cihaz: {DEVICE}, Veri Tipi: {TORCH_DTYPE}")

# Model ve işlemci global değişkenler
model = None
processor = None
model_loaded_successfully = False

# ------------------------------------
# 2. Model Yükleme Fonksiyonu (Jupyter'deki yükleme mantığına benzer)
# ------------------------------------
def load_florence_model():
    global model, processor, model_loaded_successfully
    if model_loaded_successfully: # Zaten başarıyla yüklendiyse tekrar deneme
        print("Model zaten başarıyla yüklendi.")
        return
    if model is not None: # Yükleme denenmiş ama başarısız olmuş olabilir, yine de tekrar deneme (eğer flag false ise)
        print("Model yükleme daha önce denendi, tekrar denenmiyor (başarısız olmuş olabilir).")
        return


    print(f"'{MODEL_NAME}' modeli ve işlemcisi yükleniyor...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=TORCH_DTYPE,
            trust_remote_code=True  # Önemli!
        ).to(DEVICE)
        model.eval()  # Inference modu için

        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True  # Önemli!
        )
        model_loaded_successfully = True
        print("Model ve işlemci başarıyla yüklendi.")
    except Exception as e:
        model_loaded_successfully = False
        model = None # Hata durumunda modeli None yapalım
        processor = None # Hata durumunda işlemciyi None yapalım
        print(f"HATA: Model yüklenirken bir sorun oluştu: {e}")
        # Bu hata PythonAnywhere loglarında görünecek.

# ------------------------------------
# 3. Inference Fonksiyonu (Jupyter Notebook'unuzdaki `inference` fonksiyonu)
#    - Girdileri ve çıktıları API'ye uygun hale getirildi.
#    - Global 'model' ve 'processor' kullanır.
#    - 'device' ve 'torch_dtype' API için sabitlenmiş değerleri kullanır.
# ------------------------------------
def run_inference_from_notebook(image_input, task_prompt_str, text_input_str=None): # Fonksiyon adını değiştirdim karışmaması için
    """
    Performs inference using the given image and task prompt.
    (Jupyter Notebook'tan adapte edildi)
    """
    if not model_loaded_successfully or model is None or processor is None:
        return {"error": "Model yüklenemedi veya mevcut değil."}

    # Combine task prompt with additional text input if provided
    prompt = task_prompt_str if text_input_str is None else task_prompt_str + text_input_str

    try:
        # Generate input data for model processing from the given prompt and image
        inputs = processor(
            text=prompt,
            images=image_input,
            return_tensors="pt",
        ).to(DEVICE, TORCH_DTYPE) # Sabitlenmiş DEVICE ve TORCH_DTYPE kullanılıyor

        # Generate model predictions (token IDs)
        # .cuda() çağrıları kaldırıldı, çünkü DEVICE zaten ayarlandı.
        generated_ids = model.generate(
            input_ids=inputs["input_ids"], # .to(DEVICE) zaten yukarıda yapıldı
            pixel_values=inputs["pixel_values"], # .to(DEVICE) zaten yukarıda yapıldı
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        # Decode generated token IDs into text
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=False, # Post-processing için önemli olabilir
        )[0]

        # Post-process the generated text into a structured response
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt_str, # Görev tipi post-processing için önemli
            image_size=(image_input.width, image_input.height),
        )
        return parsed_answer
    except Exception as e:
        print(f"Inference sırasında hata (run_inference_from_notebook): {e}")
        return {"error": f"Çıkarım yapılırken bir hata oluştu: {e}"}

# ------------------------------------
# 4. API Endpoint'leri
# ------------------------------------
@app.route('/')
def home():
    if not model_loaded_successfully and model is None: # Henüz yükleme denenmediyse veya başarısız olduysa
        #load_florence_model()
    status = "Model Yüklendi" if model_loaded_successfully else "Model Yüklenemedi veya Henüz Yüklenmedi"
    return f"Florence-2 API (Jupyter Entegrasyonu) Hoş Geldiniz! Model Durumu: {status}"

@app.route('/generate_caption', methods=['POST'])
def generate_caption_endpoint():
    if not model_loaded_successfully and model is None:
        #load_florence_model() # İlk istekte modeli yüklemeyi dene

    if not model_loaded_successfully:
        return jsonify({"error": "Florence-2 modeli kullanılamıyor. Sunucu loglarını kontrol edin."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "Görsel dosyası ('image') bulunamadı."}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "Dosya seçilmedi."}), 400

    try:
        image_bytes = image_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Görsel işlenirken hata: {e}"}), 400

    # Jupyter'deki task_prompt değişkenine benzer bir mantık:
    # Swift'ten 'task_prompt' form verisi olarak gönderilebilir.
    # Gönderilmezse, Jupyter'deki gibi bir varsayılan kullanabiliriz.
    # Jupyter Notebook'unuzda task_prompt = "<DETAILxED_CAPTION>" olarak tanımlı.
    # Eğer Swift'ten farklı bir prompt göndermek isterseniz, onu kullanır.
    default_task_prompt_from_notebook = "<DETAILED_CAPTION>" # Jupyter'deki 'task_prompt' (Düzeltilmiş hali)
    # Not: Jupyter'de "<DETAILxED_CAPTION>" yazmışsınız, 'x' fazla gibi duruyor.
    #      Eğer bu kasıtlı bir prompt ise, onu kullanın. Ben "<DETAILED_CAPTION>" olarak düzelttim.
    task_prompt_from_request = request.form.get('task_prompt', default_task_prompt_from_notebook)

    # text_input parametresi şu an için kullanılmıyor, None olarak geçiliyor.
    # İsterseniz bunu da Swift'ten alabilirsiniz.
    text_input_from_request = request.form.get('text_input', None)

    inference_result = run_inference_from_notebook(pil_image, task_prompt_from_request, text_input_from_request)

    if "error" in inference_result:
        return jsonify(inference_result), 500

    # Çıktıyı alma mantığı (bir önceki app.py'deki gibi, daha esnek)
    caption_output = inference_result.get(task_prompt_from_request)

    if caption_output is None:
        if isinstance(inference_result, dict) and len(inference_result) == 1:
            caption_output = list(inference_result.values())[0]
        else:
            return jsonify({"error": f"'{task_prompt_from_request}' için beklenen çıktı formatı alınamadı. Alınan: {inference_result}"}), 500

    if isinstance(caption_output, list):
        caption_text = caption_output[0] if len(caption_output) > 0 else "Model boş bir liste döndürdü."
    elif isinstance(caption_output, str):
        caption_text = caption_output
    else:
        caption_text = str(caption_output)

    cleaned_caption = caption_text.strip()
    return jsonify({"poster_caption": cleaned_caption})

# ------------------------------------
# 5. Ana Çalıştırma Bloğu (PythonAnywhere için GEREKLİ DEĞİL)
# ------------------------------------
# Bu blok sadece lokal testler için: `python app.py`
# if __name__ == '__main__':
#     print("Flask geliştirme sunucusu (LOKAL TEST - Jupyter Entegrasyonu) başlatılıyor...")
#     load_florence_model() # Lokal test için modeli hemen yükle
#     app.run(host='0.0.0.0', port=5000, debug=False)
