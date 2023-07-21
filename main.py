import streamlit as st
import os
import pinecone
import fitz

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import PyMuPDFLoader


embeddings = OpenAIEmbeddings()
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="us-central1-gcp")    

st.set_page_config(page_title="Ask PDF", page_icon=":question:")

st.header(body="Ask PDF")

st.markdown("#### How to \n 1. Upload document \n 2. Click to embed document and wait for processing (must be done only once) \n 3. Type and press Ctrl + enter or click the 'Press Ctrt + enter to apply' text)")

user_pdf = st.file_uploader(label="Upload PDF here", type="pdf")

button_click = False

if user_pdf:
    button_click = st.button(label="Click to embed document")

if button_click:
    doc = fitz.open(stream=user_pdf.getvalue(), filetype='pdf')
    doc.save('user_pdf.pdf')
    loader = PyMuPDFLoader('user_pdf.pdf')
    data = loader.load()
    texts = [doc.page_content for doc in data if doc.page_content != '']
    metadatas = [{"source": "page " + str(doc.metadata["page"])} for doc in data if doc.page_content != '']

    index = pinecone.Index("langchain-pdf-index")
    index.delete(delete_all=True)
    
    Pinecone.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        index_name="langchain-pdf-index"
    )
    

question = st.text_area(label="Your question:", placeholder="Your question...", key="question")

llm_on = True

output = None

if question and user_pdf:
    if llm_on:
        docsearch = Pinecone.from_existing_index (
            embedding=embeddings,
            index_name="langchain-pdf-index"
        )
        
        qa_sources_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0),
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={"k": 4})
        )

        output = qa_sources_chain(question, return_only_outputs=True)
    else:
        output = {'answer' : 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.',
                'sources' : 'Page 1'}

    st.markdown("#### Answer")
    st.write(output['answer'])

    st.markdown("#### Sources")
    st.write(output['sources'])
