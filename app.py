def map_label(label):
    label = label.lower()

    if "bottle" in label:
        return "Flasche"
    elif "pen" in label:
        return "Stift"
    elif "box" in label:
        return "Brotdose"
    else:
        return "Unbekannt"import streamlit as st
from transformers import pipeline
from PIL import Image

# ----------------------------
# KI Modell laden (Hugging Face)
# ----------------------------
classifier = pipeline("image-classification")

st.title("📦 Digitales Fundbüro (KI mit Hugging Face)")

# ----------------------------
# Bild hochladen
# ----------------------------
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Dein Bild", use_column_width=True)

    st.write("🔍 Analysiere Bild...")

    result = classifier(image)mapped = map_label(result[0]['label'])
st.write("➡️ Erkannt als:", mapped)
    

    # Top 3 Ergebnisse anzeigen
    for r in result[:3]:
        st.write(f"**{r['label']}** – {round(r['score']*100,2)}%")
