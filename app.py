# Step 1: Transcribe the audio files

from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import streamlit as st
# Set the working dir as the relative file path
OPENAI_API_KEY = st.secrets["OPENAI_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

# Set the prompt for the report.
TEXT_PROMPT = """You are an exceptional reporter.

### Task
You will be provided with a transcription or some audio messages. Your task is to process the transcriptions, returning the information on a specific format.

### Expected output format
The audio messages are from volunteers from "Fundación Sí", an NGO that does "recorridas", handing out food and talking to homeless people. The audio messages are reports of those "recorridas", saying who they saw, what they talked about, and who they didn't see.
The written report should be an itemization on each person mentioned, re-organizing the information that is said in the audio messages.
Make sure to take special note of anything important, leaving out anything not really important.
Make sure to write the report in Argentinian Spanish!

Transcription: {transcription}

report: """

# Step 1: Transcribe the audio files
def transcribe_audio(audio_file_path):
    audio_file = open(audio_file_path, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        language='es',
        file=audio_file, 
        response_format="text"
        )
    return transcript

# Step 2: Get the report with GPT-3.5

def get_report(transcript):
    prompt = PromptTemplate(
        template=TEXT_PROMPT,
        input_variables=['transcription']
    )
    chain = LLMChain(
        llm=ChatOpenAI(
            api_key=OPENAI_API_KEY, model='gpt-3.5-turbo', max_tokens=1000
        ), prompt=prompt, verbose=True
    )
    report = chain.predict(transcription=transcript)
    print(report)
    return report

# Step 3: Build a streamlit app.
def main():
    st.title("Transcription and Report Generation")
    st.write("This app transcribes audio files and generates a report based on the transcription.")
    audio_files = st.file_uploader("Upload all audio files for the report, in order.", type=["mp3", "flac"], accept_multiple_files=True)
    # Create a button to transcribe the audio files
    transcribe_button = st.button("Done uploading files! Transcribe and Generate Report.")
    st.session_state.transcripts = []
    if transcribe_button:
        for i, audio_file in enumerate(audio_files):
            # Save audio file
            with open(audio_file.name, "wb") as f:
                f.write(audio_file.read())
            with st.spinner(f'Transcribing audio {i + 1}...'):
                transcript = transcribe_audio(audio_file.name)
                st.session_state.transcripts.append(transcript)
        joined_transcripts = "\n\n".join(st.session_state.transcripts)
        with st.expander('Show Transcription'):
            st.write("Transcription:", joined_transcripts)
        with st.spinner('Writing report...'):
            report = get_report(joined_transcripts)
        with st.expander('Show Report'):
            st.write("Report:\n", report)

if __name__ == "__main__":
    main()