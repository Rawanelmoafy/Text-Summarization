import streamlit as st
from PIL import Image
import pytesseract
import easyocr
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_path = r"G:\Text_summarization\best_t5_model\best_t5_model"  
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)
model.eval()


# Summarize
import re
def clean_text(text):
    text = re.split(r'\b(references|appendix|acknowledgements)\b', text, flags=re.IGNORECASE)[0]
    text = re.sub(r'\\begin{.*?}|\\end{.*?}', '', text)
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    cleaned_words = [words[0]] if words else []
    for word in words[1:]:
        if word != cleaned_words[-1]:
            cleaned_words.append(word)

    final_words = []
    for word in cleaned_words:
        if not re.fullmatch(r'([a-zA-Z])\1{2,}', word):
            final_words.append(word)

    return ' '.join(final_words)

# تقسيم النص إلى chunks
def chunk_text(text, tokenizer, max_length=512):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

# تلخيص النصوص المجزأة
def summarize_chunks(text, max_words=512):
    text = clean_text(text)
    chunks = chunk_text(text, tokenizer, max_length=max_words)
    summaries = []

    for chunk in chunks:
        input_text = "abstract: " + chunk
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            summary_ids = model.generate(
                         inputs,
                         max_length=100,
                         min_length=30,
                         num_beams=4,
                         do_sample=True,
                         top_k=50,
                         top_p=0.95,
                         length_penalty=1.0,
                         early_stopping=True
)

        print("Chunk before summarization:", chunk)

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    print("Summary:", summary)

    return "\n".join(summaries)

# واجهة Streamlit
st.set_page_config(page_title="OCR/Text Summarizer", layout="centered")
st.title("🧠 OCR/Text Summarizer")
st.write("Choose to input plain text or upload an image.")

# اختيار نوع الإدخال
option = st.radio("Select input method:", ("✍️ Enter Text", "📷 Upload Image"))

if option == "✍️ Enter Text":
    user_input = st.text_area("Write or paste your text below:")
    if st.button("📝 Summarize Text"):
        if user_input.strip():
            with st.spinner("Summarizing..."):
                summary = summarize_chunks(user_input)
            st.subheader("🧾 Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text.")

elif option == "📷 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("📝 Extract & Summarize"):
            with st.spinner("Extracting text..."):
                image_np = np.array(image)
                reader = easyocr.Reader(['en', 'ar'])
                results = reader.readtext(image_np, detail=0)
                text = " ".join(results)

            st.subheader("📜 Extracted Text:")
            st.write(text)

            with st.spinner("Summarizing..."):
                summary = summarize_chunks(text)

            st.subheader("🧾 Summary:")
            st.write(summary)
