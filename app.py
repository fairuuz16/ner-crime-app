import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load model dan tokenizer dari Hugging Face
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("fairuuz/ner-crime-indobertweet")
    model = AutoModelForTokenClassification.from_pretrained("fairuuz/ner-crime-indobertweet")
    ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner_pipe

ner = load_model()

# UI Streamlit
st.title("NER Berita Kriminalitas 🇮🇩")
st.markdown("Model NER untuk ekstraksi entitas kriminal dari berita berbahasa Indonesia.")

text_input = st.text_area("Masukkan teks berita:", height=200)

if st.button("Deteksi Entitas"):
    if text_input:
        entities = ner(text_input)
        for ent in entities:
            st.markdown(
                f"- **{ent['entity_group']}**: `{ent['word']}` (score: {ent['score']:.2f})"
            )
    else:
        st.warning("Masukkan teks terlebih dahulu.")
