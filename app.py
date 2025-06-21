import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re

@st.cache_resource
def load_model():
    """Memuat pipeline NER dari HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained("fairuuz/ner-crime-bertbase")
    model = AutoModelForTokenClassification.from_pretrained("fairuuz/ner-crime-bertbase")
    ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner_pipe

def get_entity_color(entity_type, theme='Light'):
    """Mengembalikan warna yang berbeda untuk setiap jenis entitas, dengan dukungan tema."""
    light_colors = {
        'LOC': '#D5F5E3',       # Hijau Muda
        'NOR': '#D6EAF8',       # Biru Muda
        'LAW': '#FDEDEC',       # Pink Muda
        'DAT': '#FEF9E7',       # Kuning Muda
        'PER': '#E8F8F5',       # Teal Muda
        'CRIMETYPE': '#FADBD8', # Merah Muda
        'EVIDENCE': '#EBDEF0',  # Ungu Muda
    }
    dark_colors = {
        'LOC': '#2ECC71',       # Emerald
        'NOR': '#3498DB',       # Peter River
        'LAW': '#E74C3C',       # Alizarin
        'DAT': '#F1C40F',       # Sun Flower
        'PER': '#1ABC9C',       # Turquoise
        'CRIMETYPE': '#E67E22', # Carrot
        'EVIDENCE': '#9B59B6',  # Amethyst
    }
    
    if theme == 'Dark':
        return dark_colors.get(entity_type, '#7F8C8D')  
    return light_colors.get(entity_type, '#F2F3F4')   

def highlight_entities(text, entities, theme='Light'):
    """Membuat string HTML dengan entitas yang disorot menggunakan styling kustom."""
    if not entities:
        return text
    
    highlight_text_color = "black" if theme == 'Light' else "white"
    
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    highlighted_text = text
    for entity in sorted_entities:
        start = entity['start']
        end = entity['end']
        word = entity['word']
        entity_type = entity['entity_group']
        color = get_entity_color(entity_type, theme)
        
        replacement = (
            f'<span style="background-color: {color}; color: {highlight_text_color}; padding: 0.3em 0.5em; margin: 0 0.2em; line-height: 1; border-radius: 0.35em;" '
            f'title="{entity_type} (Score: {entity["score"]:.2f})">'
            f'{word}'
            f' <strong style="font-size: 0.8em; font-weight: bold;">{entity_type}</strong>'
            '</span>'
        )
        highlighted_text = highlighted_text[:start] + replacement + highlighted_text[end:]
    
    return highlighted_text

st.set_page_config(
    page_title="NER Analisis Kriminalitas",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="auto"
)

st.sidebar.title("Pengaturan Tampilan")
theme = st.sidebar.radio("Pilih Tema Aplikasi", ["Light", "Dark"])

if theme == 'Dark':
    st.markdown("""
        <style>
        /* Mengubah latar belakang utama dan warna teks */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        /* [FIX] Menghapus background putih pada header */
        [data-testid="stHeader"] {
            background-color: #0E1117;
        }
        /* Mengubah warna teks judul */
        h1, h2, h3, h4, h5, h6 {
            color: #FAFAFA;
        }
        /* Mengubah latar belakang text_area */
        .stTextArea textarea {
            background-color: #161A21;
            color: #FAFAFA;
        }
        /* [FIX] Mengubah warna teks di dalam semua alert (info, warning, error, success) */
        [data-testid="stAlert"] {
            color: #FAFAFA;
        }
        </style>
    """, unsafe_allow_html=True)


try:
    ner = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error memuat model: {str(e)}")

st.title("NER Analisis Berita Kriminalitas 🔍")
st.markdown("Ekstraksi Informasi Kriminalitas dari Teks Berita Indonesia dengan Named Entity Recognition (NER)")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Input Teks Berita")
    
    sample_text = "Pada tanggal 15 Januari 2024, Kepolisian Resor Jakarta Selatan menangkap tersangka bernama Ahmad Sutrisno di kawasan Kemang. Tersangka diduga melakukan tindak pidana pencurian dengan kekerasan menggunakan pisau lipat terhadap korban di Jalan Sudirman. Barang bukti yang diamankan berupa uang tunai Rp 2.500.000 dan handphone Samsung Galaxy."
    
    text_input = st.text_area(
        "Masukkan atau tempel teks berita di sini:",
        value=sample_text,
        height=250,
        label_visibility="collapsed"
    )
    
    analyze_button = st.button("Analisis Entitas", type="primary", use_container_width=True)

    st.subheader("Daftar Entitas")
    entity_types = {'LOC': 'Lokasi', 'NOR': 'Organisasi', 'LAW': 'Hukum', 'DAT': 'Tanggal/Waktu', 'PER': 'Person', 'CRIMETYPE': 'Jenis Kejahatan', 'EVIDENCE': 'Barang Bukti'}
    legend_html = ""
    legend_text_color = "black" if theme == 'Light' else "white"
    for etype, desc in entity_types.items():
        color = get_entity_color(etype, theme)
        border_color = "#E0E0E0" if theme == 'Light' else "#444"
        legend_html += f'<span style="background-color: {color}; color: {legend_text_color}; padding: 0.4rem 0.8rem; margin: 5px 5px 5px 0; display: inline-block; border-radius: 8px; border: 1px solid {border_color};">{desc} ({etype})</span>'
    st.markdown(legend_html, unsafe_allow_html=True)


with col2:
    st.subheader("Hasil Analisis")
    result_bg_color = "#F8F9F9" if theme == 'Light' else "#1C2833" 
    result_border_color = "#E0E0E0" if theme == 'Light' else "#31333F"
    
    if model_loaded and analyze_button and text_input.strip():
        with st.spinner("Menganalisis teks..."):
            try:
                entities = ner(text_input)
                
                if entities:
                    highlighted = highlight_entities(text_input, entities, theme)
                    st.markdown(f'<div style="line-height: 2.2; font-size: 1.1rem; padding: 1.5rem; background-color: {result_bg_color}; border-radius: 8px; border: 1px solid {result_border_color};">{highlighted}</div>', unsafe_allow_html=True)
                    
                    st.markdown("<h5 style='margin-top: 2rem;'>Daftar Entitas Terdeteksi</h5>", unsafe_allow_html=True)
                    
                    entity_groups = {}
                    for ent in entities:
                        entity_type = ent['entity_group']
                        if entity_type not in entity_groups:
                            entity_groups[entity_type] = []
                        entity_groups[entity_type].append(ent)
                    
                    for entity_type, group_entities in sorted(entity_groups.items()):
                        header_color = get_entity_color(entity_type, theme)
                        header_text_color = "black" if theme == 'Light' else "white"
                        
                        st.markdown(
                            f'<div style="background-color: {header_color}; color: {header_text_color}; padding: 0.5rem 1rem; margin-top: 1rem; border-radius: 5px; font-weight: bold;">{entity_types.get(entity_type, entity_type)} ({entity_type})</div>',
                            unsafe_allow_html=True
                        )
                        for ent in sorted(group_entities, key=lambda x: x['score'], reverse=True):
                            score = ent['score']
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- **{ent['word']}** (Score: {score:.3f})")
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
