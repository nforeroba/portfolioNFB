import yaml
import streamlit as st
import subprocess
import os
from tempfile import NamedTemporaryFile
import shutil

# Importaciones de LangChain para procesamiento de documentos y generación de resúmenes
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Importaciones de modelos de lenguaje
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

def generate_pdf_with_quarto(markdown_text):
    # Genera un archivo PDF a partir del texto Markdown usando Quarto
    with NamedTemporaryFile(delete=False, suffix=".qmd", mode='w') as md_file:
        md_file.write(markdown_text)
        md_file_path = md_file.name

    pdf_file_path = md_file_path.replace('.qmd', '.pdf')
    
    # Ejecuta el comando de Quarto para renderizar el PDF
    subprocess.run(["quarto", "render", md_file_path, "--to", "pdf"], check=True)
    
    os.remove(md_file_path)
    return pdf_file_path

def move_file_to_downloads(pdf_file_path):
    # Mueve el archivo PDF generado a la carpeta de Descargas del usuario
    downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
    destination_path = os.path.join(downloads_path, os.path.basename(pdf_file_path))
    shutil.move(pdf_file_path, destination_path)
    return destination_path

def load_and_summarize(file):
    # Carga el archivo PDF y genera un resumen utilizando un modelo de lenguaje
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        file_path = tmp.name
    
    try:
        # Carga el PDF utilizando PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Define el template para el prompt que se usará en la generación del resumen
        prompt_template = """
        Escribe un informe detallado basado en el siguiente documento:
        {text}

        Utiliza el siguiente formato en Markdown:
        # Título Descriptivo del Informe

        ## Resumen Ejecutivo
        Utiliza de 3 a 7 puntos numerados

        ## Puntos Clave:
        Describe los aspectos más importantes discutidos en el documento. Utiliza de 3 a 5 puntos numerados.

        ## Conclusiones
        Concluye con una visión general de las implicaciones más importantes del documento.
        """
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Inicializa el modelo de lenguaje (en este caso, ChatOllama)
        model = ChatOllama(
            model="llama3.1:8b"
        )

        # Configura la cadena de procesamiento de LangChain
        llm_chain = LLMChain(llm=model, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        
        # Genera el resumen
        response = stuff_chain.invoke(docs)
    finally:
        # Limpia el archivo temporal
        os.remove(file_path)

    return response['output_text']

# Configuración de la interfaz de Streamlit
st.set_page_config(layout='wide', page_title="Resumidor de Documentos")
st.title('Resumidor de Documentos PDF')
col1, col2 = st.columns(2)

# Columna para cargar el archivo
with col1:
    st.subheader('Sube un documento PDF:')
    uploaded_file = st.file_uploader("Elige un archivo", type="pdf", key="file_uploader")
    if uploaded_file:
        summarize_flag = st.button('Resumir Documento', key="summarize_button")

# Columna para mostrar el resumen y generar el PDF
if uploaded_file and summarize_flag:
    with col2:
        with st.spinner('Resumiendo...'):
            # Genera el resumen
            summaries = load_and_summarize(uploaded_file)
            st.subheader('Resultado del Resumen:')
            st.markdown(summaries)
            
            # Genera y descarga el PDF
            pdf_file = generate_pdf_with_quarto(summaries)
            download_path = move_file_to_downloads(pdf_file)
            st.markdown(f"**PDF Descargado en tu carpeta de Descargas: {download_path}**")

else:
    with col2:
        st.write("No se ha subido ningún archivo. Por favor, sube un archivo PDF para continuar.")