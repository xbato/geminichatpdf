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
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# read all pdf files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for uploaded_file in pdf_docs:
        try:
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
            st.error(f"Error procesando PDF: {e}")
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
    print("Generando embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    print("Guardando el índice FAISS localmente...")
    vector_store.save_local("faiss_index")
    print("Índice FAISS creado y guardado.")


    
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response


def main():
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
        if st.button("Subir y Procesar"):
            if pdf_docs is not None:
                st.session_state.processing_attempted = True
                process_pdf(pdf_docs)
            else:
                st.error("Por favor, carga un archivo PDF antes de presionar 'Subir y Procesar'.")

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