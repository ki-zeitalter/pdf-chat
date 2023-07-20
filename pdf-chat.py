import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

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

def conversation_chain(embeddings):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(),
        retriever = embeddings.as_retriever(),
        memory = memory
    )
    return chain

def run():
    setup()

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None

    if "chain" not in st.session_state:
        st.session_state.chain = None

    st.set_page_config(page_title="Chat mit deinen PDF-Dateien 📘", page_icon="📘")

    st.header("Chat mit deinen PDF-Dateien 📘")

    pdf_dateien = st.file_uploader("PDF-Dateien hochladen (anschließend klicke auf 'Verarbeite Dateien')",type="pdf", accept_multiple_files=True)

    if st.button("Verarbeite Dateien"):
        text = text_extrahieren(pdf_dateien)

        text_teile = text_splitten(text)

        st.write("Gesamttext mit Länge von " + str(len(text)) + " wurde in " + str(len(text_teile)) + " Teile aufgeteilt")

        embeddings = erstelle_embeddings(text_teile)

        st.session_state.embeddings = embeddings

        st.session_state.chain = conversation_chain(embeddings)


    prompt = st.text_input("Stelle eine Frage")

    if prompt:
        antwort = st.session_state.chain({"question": prompt})
        st.write(antwort)

if __name__ == '__main__':
    run()
