import os
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import io
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
import backoff
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
import uuid  # Importante para generar identificadores únicos
import time
import threading

# Configuración básica del logging
logging.basicConfig(
    level=logging.DEBUG,  # Captura todos los niveles de debug hacia arriba
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # Guarda los logs en un archivo
    filemode='w'  # Sobrescribe el archivo de log cada vez que se inicia la app
)

# Para capturar también los logs en la consola, puedes agregar un StreamHandler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console_handler)

logging.getLogger().setLevel(logging.INFO)

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class GoogleAPIError(Exception):
    """Excepción para errores al llamar a la API de Google Generative AI."""
    pass

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_heavy_file(file_path):
    print(f"Procesando archivo: {file_path}")
    time.sleep(10)  # Simula el tiempo de procesamiento
    print(f"Archivo procesado: {file_path}")

def start_background_processing(file_path):
    thread = threading.Thread(target=process_heavy_file, args=(file_path,))
    thread.start()
    return thread

# Definir una función para borrar los archivos despues de 1 dia
def clean_old_faiss_indices():
    directory = '.'  # Directorio donde se guardan los índices FAISS
    now = time.time()
    cutoff = 86400  # 24 horas * 60 minutos * 60 segundos

    for filename in os.listdir(directory):
        if filename.startswith('faiss_index_'):  # Asegúrate de que este prefijo coincida con cómo nombras los archivos de índice
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_modified = os.path.getmtime(filepath)
                if now - file_modified > cutoff:
                    os.remove(filepath)
                    print(f'Índice FAISS eliminado: {filename}')

# Definir una función que maneje los eventos de backoff, opcional pero útil para el logging
def backoff_hdlr(details):
    logging.warning("Backing off {wait:0.1f} seconds afters {tries} tries calling function {target} with args {args} and kwargs {kwargs}".format(**details))


# Decorador para implementar reintentos con backoff exponencial en la función correcta
@backoff.on_exception(backoff.expo,
                      GoogleAPIError,  # Asegúrate de que esta excepción se lanza/captura correctamente en tu código
                      max_tries=8,
                      on_backoff=backoff_hdlr)
def call_chain_with_backoff(docs, user_question):
    chain = get_conversational_chain()
    return chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)



# read all pdf files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for uploaded_file in pdf_docs:
        try:
            logging.info("Iniciando extracción de texto del PDF.")

            # Lee el contenido del archivo cargado
            bytes_data = uploaded_file.getvalue()
            
            # Extraer texto usando PDFMiner directamente de bytes
            extracted_text = extract_text(io.BytesIO(bytes_data))

            if extracted_text.strip():
                text += extracted_text + "\n"
            else:
                # Guarda temporalmente el archivo para usar con pdf2image
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(bytes_data)
                    tmp_file_path = tmp_file.name
                
                # Convertir PDF a imágenes (una por página) y aplicar OCR
                images = convert_from_path(tmp_file_path)
                for image in images:
                    text += pytesseract.image_to_string(image) + "\n"

        except Exception as e:
            logging.error("Error procesando PDF: %s", e)
            st.error(f"Error procesando PDF: {e}")
        pass
    return text

# split text into chunks


def get_text_chunks(text):
    # Suponiendo que esta función divide el texto en chunks
    # Asegúrate de que el texto no esté vacío antes de dividirlo
    if not text.strip():
        return []  # Devuelve una lista vacía si el texto está vacío

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk


def get_conversational_chain():
    prompt_template = """
   Contesta la pregunta con la información mas completa, analizando perfectamente bien el documento que el usuario te proporcionó\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

def get_vector_store(chunks):
    try:
        logging.info("Generando embeddings y guardando el índice FAISS específico del usuario.")
        print("Generando embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        faiss_index_path = get_user_specific_faiss_index_path()
        print(f"Guardando el índice FAISS en {faiss_index_path}...")
        vector_store.save_local(faiss_index_path)
        print("Índice FAISS creado y guardado específicamente para el usuario.")
    except Exception as e:
        logging.error("Error generando embeddings o guardando el índice FAISS: %s", e)
        st.error(f"Error durante la generación de embeddings o al guardar el índice FAISS: {e}")
        st.write(f"Detalles del error: {e}")

# Función modificada para gestionar el índice FAISS por usuario
def get_user_specific_faiss_index_path():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    return f"faiss_index_{st.session_state.user_id}"

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        faiss_index_path = get_user_specific_faiss_index_path()
        new_db = FAISS.load_local(faiss_index_path, embeddings)
        docs = new_db.similarity_search(user_question)
        response = call_chain_with_backoff(docs, user_question)
        print(response)
        return response
    
    except GoogleAPIError as e:  # Asegúrate de que esta excepción es la correcta para tu caso
        logging.error("Error al llamar a Google Generative AI API: %s", e)
        st.error("Hubo un problema al conectar con Google Generative AI API. Por favor, inténtalo de nuevo.")
        return {"output_text": "Error al conectar con el servicio de Google Generative AI."}


def main():
    clean_old_faiss_indices()  # Limpieza de índices antiguos al inicio
    st.set_page_config(page_title="Tu PDF.AI", page_icon="🤖")

    # CSS para ocultar el menú hamburguesa y el pie de página "Made with Streamlit"
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    if 'pdf_preview' not in st.session_state:
        st.session_state.pdf_preview = None
    if 'processing_attempted' not in st.session_state:
        st.session_state.processing_attempted = False

    with st.sidebar:
        st.title("Menú:")
        pdf_docs = st.file_uploader("Sube tu archivo PDF y haz click en Subir y Procesar", type=["pdf"], accept_multiple_files=False)

        if pdf_docs is not None:
            # Verifica el tamaño del archivo aquí
            file_size = pdf_docs.size  # Tamaño del archivo en bytes
            file_size_mb = file_size / (1024 * 1024)  # Convertir a megabytes

            if file_size_mb > 35:
                st.error("El archivo supera el límite de 35MB. Por favor, carga un archivo más pequeño.")
            else:
                if st.button("Subir y Procesar"):
                    process_pdf(pdf_docs)

    # Main content area for displaying chat messages and PDF previews
    st.write("Conversa con tu PDF!")
    if st.session_state.pdf_preview is not None:
        st.image(st.session_state.pdf_preview, caption='Preview de la primera página del PDF', use_column_width=True)

    # Resto de la lógica para el chat y el procesamiento de preguntas
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Sube tu PDF y hazme preguntas"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''.join(response.get('output_text', ["Ocurrió un error al procesar tu pregunta. Por favor, inténtalo de nuevo."]))
                placeholder.markdown(full_response)
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)

    st.sidebar.button('Borrar Historial del Chat', on_click=clear_chat_history)

def process_pdf(pdf_docs):
    try:
        # Verificación del tamaño del archivo
        file_size = pdf_docs.size  # Tamaño del archivo en bytes
        file_size_mb = file_size / (1024 * 1024)  # Convertir a megabytes
        
        # Si el archivo es mayor a 35 MB, muestra un mensaje de error y retorna
        if file_size_mb > 35:
            st.error("El archivo supera el límite de 35MB. Por favor, carga un archivo más pequeño.")
            return
        logging.info("Procesando archivo PDF.")
    except Exception as e:
        logging.error("Error al procesar el PDF: %s", e)
        st.error(f"Error al procesar el PDF: {e}")

    try:
        bytes_data = pdf_docs.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(bytes_data)
            tmp_file_path = tmp_file.name

        images = convert_from_path(tmp_file_path)
        st.session_state.pdf_preview = images[0]
        st.success("PDF procesado con éxito.")
    except Exception as e:
        st.error(f"Error al procesar el PDF: {e}")

    raw_text = get_pdf_text([pdf_docs])
    text_chunks = get_text_chunks(raw_text)
    if text_chunks:
        get_vector_store(text_chunks)
    else:
        if st.session_state.processing_attempted:
            st.error("No se encontró texto para procesar.")


if __name__ == "__main__":
    main()