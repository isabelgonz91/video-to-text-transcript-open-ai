import os
import streamlit as st
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from dotenv import load_dotenv
import base64
import tempfile
import moviepy.editor as mp

# Cargar variables de entorno desde un archivo .env
load_dotenv()

# Configurar la API de OpenAI utilizando las variables de entorno
openai_api_key = os.getenv("OPENAI_API_KEY")

# Clase personalizada para cargar archivos de audio locales
class LocalAudioLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def yield_blobs(self):
        st.info(f"Yielding file path: {self.file_path}")
        yield FileBlob(self.file_path)

class FileBlob:
    def __init__(self, path):
        self.path = path
        self.source = "local"

# Función para extraer el audio del video cargado
def extract_audio(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = os.path.join(tempfile.gettempdir(), "extracted_audio.mp3")
    video.audio.write_audiofile(audio_path)
    return audio_path

# Función principal para la interfaz de Streamlit
def main():
    st.title("Transcripción de Audio de Video")

    api_key = st.text_input("Clave de API de OpenAI", type="password")
    uploaded_file = st.file_uploader("Carga un archivo de video", type=["mp4", "avi", "mov"])
    language = st.selectbox("Selecciona el idioma del audio", ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh"])

    if st.button("Transcribir"):
        if not api_key or not uploaded_file:
            st.error("Por favor, proporciona la clave de API de OpenAI y carga un archivo de video.")
            return

        with tempfile.NamedTemporaryFile(delete=False) as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name

        # Extraer el audio del archivo de video cargado
        audio_path = extract_audio(temp_video_path)
        st.info(f"Audio extraído: {audio_path}")

        # Crear una instancia del cargador genérico utilizando la clase LocalAudioLoader y el parser OpenAIWhisperParser
        loader = GenericLoader(LocalAudioLoader(audio_path), OpenAIWhisperParser(api_key=api_key))
        st.info(f"Loader creado con audio_path: {audio_path}")

        try:
            # Cargar los documentos (transcripciones de audio)
            docs = loader.load()
            st.info(f"Documentos cargados: {len(docs)}")

            if docs:
                for doc in docs:
                    # Crear el nombre del archivo de texto reemplazando la extensión del archivo de audio por .txt
                    text_filename = os.path.splitext(audio_path)[0] + ".txt"
                    # Guardar la transcripción en un archivo de texto
                    with open(text_filename, "w", encoding="utf-8") as text_file:
                        text_file.write(doc.page_content)
                    st.success(f"Transcripción guardada en {text_filename}")
                    st.text_area("Transcripción", doc.page_content, height=300)

                    # Crear un enlace de descarga para la transcripción
                    b64 = base64.b64encode(doc.page_content.encode()).decode()
                    href = f'<a href="data:text/plain;base64,{b64}" download="{text_filename}">Descargar Transcripción</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.error("No se cargaron documentos.")
        except Exception as e:
            st.error(f"Ocurrió un error durante la transcripción: {e}")
            st.error(str(e))  # Mostrar detalles del error

if __name__ == "__main__":
    main()
