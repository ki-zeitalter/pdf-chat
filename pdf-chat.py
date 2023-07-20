import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def setup():
    load_dotenv()

def text_extrahieren(pdfDateien):
    gesamt_text = ""

    for pdf in pdfDateien:
        reader = PdfReader(pdf)
        for page in reader.pages:
            gesamt_text += page.extract_text()
    
    return gesamt_text


def text_splitten(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap  = 50,
        length_function = len,
    )

    return splitter.split_text(text)


def erstelle_embeddings(textTeile):
    return FAISS.from_texts(texts=textTeile, embedding=OpenAIEmbeddings())

def suche(suchtext):
    ergebnisse = st.session_state.embeddings.similarity_search(suchtext)

    for ergebnis in ergebnisse:
        st.write(ergebnis.page_content)
        st.divider()


def run():
    setup()

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None

    st.set_page_config(page_title="Embeddings aus deinen PDF-Dateien 📘", page_icon="📘")

    st.header("Embeddings aus deinen PDF-Dateien 📘")

    pdf_dateien = st.file_uploader("PDF-Dateien hochladen (anschließend klicke auf 'Erstelle Embeddings')",type="pdf", accept_multiple_files=True)

    if st.button("Erstelle Embeddings"):
        text = text_extrahieren(pdf_dateien)

        text_teile = text_splitten(text)

        st.write("Gesamttext mit Länge von " + str(len(text)) + " wurde in " + str(len(text_teile)) + " Teile aufgeteilt")

        embeddings = erstelle_embeddings(text_teile)

        st.session_state.embeddings = embeddings


    suchtext = st.text_input("Suche relevante Textstellen")

    if suchtext:
        suche(suchtext)

if __name__ == '__main__':
    run()
