import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

st.set_page_config(page_title="AI-Assisted Telemedicine", page_icon="💙", layout="wide")

# ====== BLUE + BLACK DYNAMIC THEME ======
st.markdown("""
<style>
:root {
    --deep-black: #0a0a0a;
    --night-black: #121212;
    --primary-blue: #0d6efd;
    --cyan-blue: #1a9fff;
    --blue-glow: #2b9fd9;
    --light-blue: #b3d9ff;
    --text-light: #f5f9ff;
    --text-dark: #0a0a0a;
}

/* Animated background */
.stApp {
    background: linear-gradient(130deg, var(--deep-black), var(--primary-blue), var(--night-black));
    background-size: 300% 300%;
    animation: bgShift 12s ease infinite;
    font-family: 'Segoe UI', 'Inter', sans-serif;
}
@keyframes bgShift {
    0% { background-position: 0% 50% }
    50% { background-position: 100% 50% }
    100% { background-position: 0% 50% }
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--night-black) 20%, var(--primary-blue) 100%);
    box-shadow: 4px 0 15px rgba(0,0,0,0.6);
}
section[data-testid="stSidebar"] * {
    color: var(--text-light) !important;
}

/* Titles */
h1 {
    color: var(--text-light);
    font-size: 2.4rem;
    text-shadow: 0 0 10px var(--cyan-blue);
    border-left: 5px solid var(--cyan-blue);
    padding-left: 0.8rem;
}
h2, h3, .stSubheader {
    color: var(--light-blue);
    border-left: 4px solid var(--primary-blue);
    padding-left: 0.5rem;
}

/* Cards/containers */
.block-container {
    background: rgba(15,15,15,0.6);
    border-radius: 16px;
    backdrop-filter: blur(8px);
    padding: 2rem;
    color: var(--text-light);
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
}
.stMarkdown, .stWrite {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: var(--text-light);
}

/* Inputs */
textarea, select, input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid var(--cyan-blue) !important;
    color: var(--text-light) !important;
    border-radius: 6px !important;
    padding: 0.5rem 0.75rem !important;
}
textarea:focus, select:focus, input:focus {
    outline: none !important;
    border-color: var(--light-blue) !important;
    box-shadow: 0 0 8px var(--cyan-blue);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, var(--primary-blue), var(--cyan-blue));
    color: var(--text-light);
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(0,150,255,0.4);
    transition: 0.2s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg, var(--cyan-blue), var(--primary-blue));
    transform: scale(1.03);
    box-shadow: 0 0 15px var(--blue-glow);
}

/* Alert boxes */
div.stAlert {
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    font-weight: 500;
}
.stAlert[data-baseweb="alert"] {
    background-color: rgba(0,180,0,0.2);
    color: #b3ffb3;
}
[data-baseweb="alert"][kind="warning"] {
    background-color: rgba(255,200,0,0.2);
    color: #fff6b3;
}
[data-baseweb="alert"][kind="error"] {
    background-color: rgba(255,0,0,0.25);
    color: #ffb3b3;
}

/* Lists */
.remedy-list li, ul li {
    color: var(--text-light);
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--cyan-blue);
    border-radius: 10px;
    background: rgba(255,255,255,0.07);
}

/* Map container */
[data-testid="stVerticalBlock"]>div:has([class*=folium]) {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    border: 1px solid var(--cyan-blue);
}

/* Remove footer */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==== LOGIC & FEATURES (UNCHANGED) ====
@st.cache_data
def load_data():
    diseases_symptoms = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
    symptom_disease = pd.read_csv("Symptom2Disease.csv")
    medication_db = pd.read_csv("disease_to_example_medications.csv")
    natural_rem_db = pd.read_csv("disease_to_natural_remedies.csv")
    return diseases_symptoms, symptom_disease, medication_db, natural_rem_db

diseases_symptoms, symptom_disease, medication_db, natural_rem_db = load_data()
translator = Translator()

def translate_to_english(text, lang_choice):
    try:
        if lang_choice == "en": return text
        return translator.translate(text, dest='en').text
    except:
        return text

vectorizer = TfidfVectorizer()
X_symptoms = vectorizer.fit_transform(symptom_disease['symptoms'].astype(str))

def predict_diseases(text, top_n=3):
    sims = cosine_similarity(vectorizer.transform([text]), X_symptoms).flatten()
    idxs = sims.argsort()[::-1][:top_n]
    return [(symptom_disease.iloc[i]['disease'], round(sims[i]*100, 2)) for i in idxs]

def triage_msg(conf):
    if conf >= 80: return "🔥 Urgent attention"
    elif conf >= 50: return "⚠️ See a doctor soon"
    return "✅ Self-care okay"

def find_medications(disease):
    disease_norm = disease.strip().lower()
    medication_db['disease_norm'] = medication_db['disease'].astype(str).str.strip().str.lower()
    match = medication_db[medication_db['disease_norm'] == disease_norm]
    if match.empty:
        match = medication_db[medication_db['disease_norm'].str.contains(disease_norm, na=False)]
    if not match.empty:
        return {
            "drug_classes": "; ".join(match['drug_classes'].dropna().unique()) or "N/A",
            "example_drugs": "; ".join(match['example_drugs'].dropna().unique()) or "N/A",
            "key_notes": "; ".join(match['key_notes'].dropna().unique()) or ""
        }
    return {"drug_classes": "OTC / Supportive Care",
            "example_drugs": "Paracetamol; Hydration",
            "key_notes": "General supportive measures."}

def find_natural_remedies(disease):
    disease_norm = disease.strip().lower()
    natural_rem_db['disease_norm'] = natural_rem_db['disease'].astype(str).str.strip().str.lower()
    match = natural_rem_db[natural_rem_db['disease_norm'] == disease_norm]
    if match.empty:
        match = natural_rem_db[natural_rem_db['disease_norm'].str.contains(disease_norm, na=False)]
    if not match.empty:
        return match['natural_remedies'].iloc[0]
    return "Hydration; Rest; Balanced diet"

def nearby_hospitals(lat=28.6139, lon=77.2090):
    return [
        ("City Hospital", lat + 0.01, lon + 0.01),
        ("Metro Clinic", lat - 0.01, lon - 0.02),
        ("Green Valley Hospital", lat + 0.015, lon - 0.015)
    ]

# ==== SIDEBAR & MAIN ====
st.sidebar.header("Settings")
lang_choice = st.sidebar.selectbox("Language", ["en", "hi", "es", "fr"])
input_mode = st.sidebar.radio("Input mode", ["Type symptoms", "Select symptoms"])
show_map = st.sidebar.checkbox("Show nearby hospitals", True)
enable_doctor_conn = st.sidebar.checkbox("Enable doctor connection", False)

st.title("AI-Assisted Telemedicine")

if input_mode == "Type symptoms":
    symptom_text = st.text_area("Enter symptoms", height=70)
else:
    symptoms_list = [c for c in diseases_symptoms.columns if c != 'Disease']
    selected = st.multiselect("Select symptoms", symptoms_list)
    symptom_text = ", ".join(selected)

if st.button("Get Assessment"):
    if symptom_text.strip():
        sym_en = translate_to_english(symptom_text, lang_choice)
        st.session_state['predictions'] = predict_diseases(sym_en)
        st.session_state['med_choice'] = None
    else:
        st.warning("Please enter or select symptoms.")

if 'predictions' in st.session_state and st.session_state['predictions']:
    st.subheader("Assessment")
    for dis, conf in st.session_state['predictions']:
        st.markdown(
            f"<div class='stMarkdown'><b>{dis}</b> — <span>{conf:.1f}%</span> — <i>{triage_msg(conf)}</i></div>",
            unsafe_allow_html=True
        )

    primary = st.session_state['predictions'][0][0]
    col1, col2 = st.columns(2)
    if col1.button("Natural Remedies"): st.session_state['med_choice'] = "natural"
    if col2.button("Medical Remedies"): st.session_state['med_choice'] = "medical"

    if st.session_state.get('med_choice'):
        if st.session_state['med_choice'] == "natural":
            st.markdown(f"<div class='stSubheader'>Natural Remedies for {primary}</div>", unsafe_allow_html=True)
            remedies = find_natural_remedies(primary).split(";")
            st.markdown("<ul class='remedy-list'>" + "".join(f"<li>{item.strip()}</li>" for item in remedies) + "</ul>", unsafe_allow_html=True)
        else:
            info = find_medications(primary)
            st.markdown(f"<div class='stSubheader'>Medical Remedies for {primary}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stWrite'><b>Drug Classes:</b> {info['drug_classes']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stWrite'><b>Example Drugs:</b> {info['example_drugs']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stWrite'><b>Notes:</b> {info['key_notes']}</div>", unsafe_allow_html=True)
        st.caption("⚠️ Informational only — consult a doctor.")

    if show_map:
        st.subheader("Nearby Hospitals")
        lat, lon = 28.6139, 77.2090
        m = folium.Map(location=[lat, lon], zoom_start=12, tiles="cartodbpositron")
        folium.Marker([lat, lon], tooltip="You are here", icon=folium.Icon(color="blue")).add_to(m)
        for name, la, lo in nearby_hospitals(lat, lon):
            folium.Marker([la, lo], tooltip=name, icon=folium.Icon(color="lightblue")).add_to(m)
        st_folium(m, width=650, height=350)

    if enable_doctor_conn:
        st.subheader("Available Doctors")
        st.markdown("<div class='stWrite'>- Dr. A Sharma — Now</div>", unsafe_allow_html=True)
        st.markdown("<div class='stWrite'>- Dr. B Kumar — In 30 min</div>", unsafe_allow_html=True)
        if st.button("Request Callback"):
            st.success("Request sent! A doctor will contact you.")

if st.button("Clear Results"):
    st.session_state.clear()
    st.experimental_rerun()
