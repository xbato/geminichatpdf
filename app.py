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
import base64
import tempfile


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


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_base64_pdf(pdf_file):
    """Convierte un archivo PDF a string Base64."""
    with open(pdf_file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    return base64_pdf

def show_pdf(base64_pdf):
    """Muestra un PDF codificado en Base64 en un iframe."""
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

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
        {"role": "assistant", "content": "Sube tu PDF y pregúntame lo que necesites saber"}]


def user_input(user_question):
    try:
        # Asume que embeddings se obtienen de alguna parte
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore

        # Verifica si el índice ya existe o no, y carga o crea según sea necesario
        if 'faiss_index_loaded' not in st.session_state or not st.session_state.faiss_index_loaded:
            # Intenta cargar el índice local. Asegúrate de que "faiss_index" es el nombre correcto del archivo del índice.
            # Si el archivo no existe o hay un error, esta parte fallará.
            new_db = FAISS.load_local("faiss_index", embeddings)
            st.session_state.faiss_index_loaded = True
            st.session_state.new_db = new_db  # Guarda el índice cargado en el estado de la sesión si necesitas reutilizarlo
        else:
            new_db = st.session_state.new_db  # Reutiliza el índice cargado previamente

        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        print(response)
        return response

    except Exception as e:
        # Maneja el error adecuadamente. Podrías querer mostrar un mensaje en la UI con st.error()
        print(f"Error al procesar la entrada del usuario o al cargar el índice de FAISS: {e}")
        st.error("Ocurrió un error al procesar tu pregunta. Por favor, inténtalo de nuevo.")
        return None  # O una respuesta de error adecuada


# Ejemplo de generación y guardado del índice de FAISS
def generate_and_save_faiss_index():
    # Suponiendo que embeddings es tu conjunto de datos de embeddings
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "faiss_index")
    print("Índice de FAISS guardado correctamente.")

# Cargar el índice de FAISS desde una ubicación específica
def load_faiss_index(index_path="faiss_index"):
    try:
        index = faiss.read_index(index_path)
        return index
    except RuntimeError as e:
        print(f"Error al cargar el índice de FAISS: {e}")
        # Manejo adicional del error según sea necesario


def main():
    st.set_page_config(page_title="Tu PDF.AI", page_icon="🤖")

    # Inicializa el estado si es necesario
    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False

    if 'base64_pdf' not in st.session_state:
        st.session_state.base64_pdf = ""

    # Sidebar para subir archivos PDF
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Sube tu PDF y da click en el botón de subir y procesar", accept_multiple_files=False, type="pdf")
        if st.button("Subir y procesar") and pdf_docs is not None:
            st.session_state.pdf_uploaded = True
            with st.spinner("Procesando..."):
                # Convierte el PDF a Base64 para mostrarlo
                st.session_state.base64_pdf = base64.b64encode(pdf_docs.getvalue()).decode('utf-8')
                st.success("Procesado con éxito")

    # Mostrar el PDF en el sidebar después de procesar y solo si se ha subido
    if st.session_state.pdf_uploaded and st.session_state.base64_pdf:
        pdf_display = f"""<iframe src="data:application/pdf;base64,{st.session_state.base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>"""
        st.sidebar.markdown(pdf_display, unsafe_allow_html=True)

    # Área principal para mostrar mensajes del chat
    st.title("Tu PDF.AI 🤖")
    st.write("Platica con tus archivos PDFs!")
    st.sidebar.button('Borrar Historial', on_click=lambda: clear_chat_history())

    # Chat input y respuestas
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Sube tu PDF y pregúntame lo que necesites saber"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    placeholder = st.empty()
                    full_response = ''
                    for item in response['output_text']:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
                    if response is not None:
                        message = {"role": "assistant", "content": full_response}
                        st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
