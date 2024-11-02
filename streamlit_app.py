from transformers import pipeline
import streamlit as st
from PIL import Image
from io import BytesIO
import requests
from huggingface_hub import HfApi
import pandas as pd
import pycountry

HG = {
    "automatic-speech-recognition": {
        "return_timestamps": True,
    }
}


# languages = [lang.alpha_2.lower() for lang in pycountry.countries]

languages = ["vi", "en"]
task = "automatic-speech-recognition"


# speech
# pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device_map='auto', return_timestamps=HG[task].get("return_timestamps", False))
#
# data = pipe("https://www2.cs.uic.edu/~i101/SoundFiles/taunt.wav")
#
# image classifier
# url = 'https://upload.wikimedia.org/wikipedia/commons/a/a5/Glazed-Donut.jpg'  # Replace with your image URL
# response = requests.get(url)
# img = Image.open(response.content)
# classifier = pipeline("image-classification", model="microsoft/resnet-50")
# data = classifier(img)
#
#
# print(data)
st.set_page_config(page_title="You can test any model opensource in Hugging Face", page_icon="📖", layout="wide")
st.header("You can test any model opensource in Hugging Face")


with st.sidebar:
    st.markdown(
        "## How to use\n"
        "1. Select type task in HuggingFace🔑\n"  # noqa: E501
        "2. Select Model of this task📄\n"
    )
    task = st.selectbox("Select Task", ["automatic-speech-recognition", "image-classification"])
    if task is not None:
        st.session_state["task"] = task
        lang = st.selectbox("Language", languages)
        api = HfApi()
        models = api.list_models(
            task="automatic-speech-recognition",
            sort="downloads",
            language=lang,
        )
        model = st.selectbox("Select Model", [f"{model.modelId}" for model in models])
        if model is not None:
            st.session_state["model"] = model


if st.session_state.get("task") == "automatic-speech-recognition":

    uploaded_file = st.file_uploader(
        "Upload file audio",
        type=["wav", "mp3"],
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("Predict"):

            pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device_map='auto', return_timestamps=HG[task].get("return_timestamps", False))

            data = pipe(uploaded_file.read())
            print(data)
            df = pd.DataFrame(
                {
                    "text": [d["text"] for d in data['chunks']],
                    "start->end": [d["timestamp"] for d in data['chunks']],
                }
            )
            st.table(df)
