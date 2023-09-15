import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import openai
openai.openai_api_key="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"
with st.sidebar:
    st.title("😊 ☁️Moses LLM Chat App")
    st.markdown(
        """ 
        ## About
        This App is an LLM-powered chatbot build using
        - [Streamlit](https://streamlit.io)
        - [Langchain](https://python.langchain.com/)
        
        
        """
    )
    add_vertical_space(5)
    st.write("Model with ❤️ by Moses")



def main():
    load_dotenv()
    st.write("Chat with PDF ☁️")
    pdf = st.file_uploader("upload your pdf", type="pdf")
    # st.write(pdf.name)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        print(vectorstore)
if __name__ == '__main__':

    main()