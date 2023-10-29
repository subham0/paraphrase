import streamlit as st
from paraphrasing_models import ParaphrasingModels

# Initialize the paraphrasing models
if "paraphrasing_models" not in st.session_state:
    st.session_state.paraphrasing_models = ParaphrasingModels()

st.title("Module 2 - Paraphrasing Admin Questions")

# Input text box for the user to input the text they want to paraphrase
text_to_paraphrase = st.text_area("Enter the text you want to paraphrase:")

# Dropdown for the user to select the paraphrasing model
model_choice = st.selectbox("Choose a Paraphrasing Model:", ["tuner007/pegasus_paraphrase", "Vamsi/T5_Paraphrase_Paws"])

# Dropdown for the user to select the number of paraphrases
num_paraphrases = st.selectbox("Choose the number of paraphrases:", list(range(1, 11)))

# Button to generate paraphrases
if st.button("Generate Paraphrase"):
    with st.spinner('Generating paraphrases...'):
        paraphrases = st.session_state.paraphrasing_models.paraphrase_text(text_to_paraphrase, model_choice, num_paraphrases)
        for idx, paraphrase in enumerate(paraphrases, 1):
            st.write(f"Paraphrase {idx}: {paraphrase}")

st.write("Note: The paraphrasing might take a few seconds depending on the model and the length of the text.")
