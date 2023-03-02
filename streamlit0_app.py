%%writefile app.py

import io
import os
import streamlit as st
from google.cloud import speech
import openai

# Set up the OpenAI API key
openai.api_key = "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"

# Set up the Google Cloud Speech-to-Text API
client = speech.SpeechClient()

# Define the streamlit app
st.set_page_config(page_title="Voice Assistant", page_icon="🎤", layout="wide")

# Set up the audio recording widget
audio_file = st.file_uploader("Upload audio", type=["mp3", "wav"])
if audio_file:
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")

    # Transcribe the audio to text
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US'
    )
    response = client.recognize(config=config, audio=audio)
    transcript = response.results[0].alternatives[0].transcript
    st.write("You said:", transcript)

    # Generate a response using GPT-3 API
    prompt = "User: {}\nAI:".format(transcript)
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
    )
    ai_response = response.choices[0].text.strip()
    st.write("AI:", ai_response)
