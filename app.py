import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Loads the NER pipeline from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained("fairuuz/ner-crime-indobertweet")
    model = AutoModelForTokenClassification.from_pretrained("fairuuz/ner-crime-indobertweet")
    ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner_pipe

# --- STYLING & UI HELPERS ---

def get_entity_color(entity_type):
    """Return a distinct color for each entity type."""
    # Colors adjusted for better visibility on a light theme
    colors = {
        'LOC': '#D5F5E3', # Light Green
        'NOR': '#D6EAF8', # Light Blue
        'LAW': '#FDEDEC', # Light Pink/Red
        'DAT': '#FEF9E7', # Light Yellow
        'PER': '#E8F8F5', # Light Teal
        'CRIMETYPE': '#FDEBD0', # Light Orange
        'EVIDENCE': '#EBDEF0',  # Light Purple
    }
    return colors.get(entity_type, '#F2F3F4')  # Default light gray

def highlight_entities(text, entities):
    """Creates an HTML string with entities highlighted using custom styling."""
    if not entities:
        return text
    
    # Sort entities by start position (descending) to avoid index issues
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    highlighted_text = text
    for entity in sorted_entities:
        start = entity['start']
        end = entity['end']
        word = entity['word']
        entity_type = entity['entity_group']
        color = get_entity_color(entity_type)
        
        # Simple span with background color and a tooltip
        replacement = (
            f'<span style="background-color: {color}; padding: 0.3em 0.5em; margin: 0 0.2em; line-height: 1; border-radius: 0.35em;" '
            f'title="{entity_type} (Score: {entity["score"]:.2f})">'
            f'{word}'
            f' <strong style="font-size: 0.8em; font-weight: bold;">{entity_type}</strong>'
            '</span>'
        )
        highlighted_text = highlighted_text[:start] + replacement + highlighted_text[end:]
    
    return highlighted_text

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NER Analisis Kriminalitas",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- APP LAYOUT ---

# Load Model
try:
    ner = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {str(e)}")

# Header Section
st.title("NER Analisis Berita Kriminalitas 🔍")
st.markdown("Ekstraksi Informasi Kriminalitas dari Teks Berita dengan Named Entity Recognition (NER)")

# Main container with two columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # Input Section
    st.subheader("Input Teks Berita")
    
    sample_text = "Pada tanggal 15 Januari 2024, Kepolisian Resor Jakarta Selatan menangkap tersangka bernama Ahmad Sutrisno di kawasan Kemang. Tersangka diduga melakukan tindak pidana pencurian dengan kekerasan menggunakan pisau lipat terhadap korban di Jalan Sudirman. Barang bukti yang diamankan berupa uang tunai Rp 2.500.000 dan handphone Samsung Galaxy."
    
    text_input = st.text_area(
        "Masukkan atau tempel teks berita di sini:",
        value=sample_text,
        height=250,
        label_visibility="collapsed"
    )
    
    analyze_button = st.button("Analisis Entitas", type="primary", use_container_width=True)

    # Entity Legend
    st.subheader("Daftar Entitas")
    entity_types = {'LOC': 'Lokasi', 'NOR': 'Organisasi', 'LAW': 'Hukum', 'DAT': 'Tanggal/Waktu', 'PER': 'Person', 'CRIMETYPE': 'Jenis Kejahatan', 'EVIDENCE': 'Barang Bukti'}
    legend_html = ""
    for etype, desc in entity_types.items():
        color = get_entity_color(etype)
        legend_html += f'<span style="background-color: {color}; padding: 0.4rem 0.8rem; margin: 5px 5px 5px 0; display: inline-block; border-radius: 8px; border: 1px solid #E0E0E0;">{desc} ({etype})</span>'
    st.markdown(legend_html, unsafe_allow_html=True)


with col2:
    # Results Section
    st.subheader("Hasil Analisis")
    if model_loaded and analyze_button and text_input.strip():
        with st.spinner("Menganalisis teks..."):
            try:
                entities = ner(text_input)
                
                if entities:
                    # Display highlighted text
                    highlighted = highlight_entities(text_input, entities)
                    st.markdown(f'<div style="line-height: 2.2; font-size: 1.1rem; padding: 1.5rem; background-color: #F8F9F9; border-radius: 8px; border: 1px solid #E0E0E0;">{highlighted}</div>', unsafe_allow_html=True)
                    
                    # Display entity list, grouped by type
                    st.markdown("<h5 style='margin-top: 2rem;'>Daftar Entitas Terdeteksi</h5>", unsafe_allow_html=True)
                    entity_groups = {}
                    for ent in entities:
                        entity_type = ent['entity_group']
                        if entity_type not in entity_groups:
                            entity_groups[entity_type] = []
                        entity_groups[entity_type].append(ent)
                    
                    for entity_type, group_entities in sorted(entity_groups.items()):
                        st.markdown(f"**{entity_type}**")
                        for ent in sorted(group_entities, key=lambda x: x['score'], reverse=True):
                            score = ent['score']
                            st.markdown(f"- **{ent['word']}** (Score: {score:.3f})")
                else:
                    st.info("Tidak ada entitas yang terdeteksi dalam teks yang diberikan.")
                
            except Exception as e:
                st.error(f"Terjadi error saat analisis: {str(e)}")

    elif analyze_button and not text_input.strip():
        st.warning("Harap masukkan teks terlebih dahulu untuk dianalisis.")
    else:
        st.info("Hasil analisis akan ditampilkan di sini setelah Anda menekan tombol 'Analisis Entitas'.")

if not model_loaded:
    st.error("Model gagal dimuat. Mohon refresh halaman untuk mencoba lagi.")
