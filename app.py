# Step 1: Transcribe the audio files

from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import streamlit as st

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ.get('OPENAI_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

# Step 1: Transcribe the audio files
def transcribe_audio(audio_file):
    audio_file = open("testing.mp3", "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file, 
        response_format="text"
        )
    return transcript

# Step 2: Get the report with GPT-3.5

def get_report(transcript):
    with open('instructions_V0.md', 'r') as f:
        prompt_text = f.read()
    print(prompt_text)
    prompt = PromptTemplate(
        template=prompt_text,
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
# transcript_test = "Bueno, pasamos por lo de Claudio y nos encontramos con otro chico que no conocíamos que se llama Fede Nos contó que Claudio está trabajando, creo que hace un horario de 9 a 1 y media, 2 de la mañana Está en la posición de bachero y contaba este chico Federico que se estaba quedando con ellos porque también le habían conseguido un puesto de trabajo en otro Kentucky por floresta Este chico Federico debe rondar los 30 años, 30 y algo de años Contó que hace bastante que está en situación de calle y básicamente porque se fue de su casa Nos contó que le encantan los animales y que se juntó con Claudio y con Sergio porque Sergio lo conoce del lugar donde se van a bañar ¿Cómo se llama el lugar? ¿Ahí tú te acordás? Bueno, que no lo recordamos Él antes de parar acá, paraba en el seat campeador."
# report = get_report(transcript_test)

# Step 3: Build a streamlit app.
def main():
    st.title("Transcription and Report Generation")
    st.write("This app transcribes audio files and generates a report based on the transcription.")
    audio_files = st.file_uploader("Upload all audio files for the report, in order.", type=["mp3", "flac"], accept_multiple_files=True)
    # Create a button to transcribe the audio files
    transcribe_button = st.button("Done uploading files! Transcribe and Generate Report.")
    if transcribe_button:
        transcripts = []
        for i, audio_file in enumerate(audio_files):
            with st.spinner(f'Transcribing audio {i}...'):
                transcript = transcribe_audio(audio_file)
                transcripts.append(transcript)
        joined_transcripts = "\n\n".join(transcripts)
        with st.expander('Show Transcription'):
            st.write("Transcription:", joined_transcripts)
        with st.spinner('Writing report...'):
            report = get_report(transcript)
        with st.expander('Show Report'):
            st.write("Report:", report)

if __name__ == "__main__":
    main()