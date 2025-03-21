import requests
import wikipediaapi
from bs4 import BeautifulSoup as bs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from keybert import KeyBERT
import re


kw_model = KeyBERT()


embeddings_model = AzureOpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key="BDMx6AtLJ16I2FzrSvKtDrccbbGotk9hacpJCSHKLsK6Z4Zo4Q1gJQQJ99BAACHYHv6XJ3w3AAAAACOGjqmK",
        openai_api_base="https://ai-elyeshajji2665ai524484593928.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15",
        openai_api_version="2023-05-15",) 



def get_vectorstore(question):
    all_docs=[]
    keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 8), stop_words="english", top_n=20)
    titles = [kw[0] for kw in keywords]
    print(titles)

    if titles==[]:
      return False
    wiki=wikipediaapi.Wikipedia('llms (elyes.hajj)','en')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,length_function=len,)
    for title in titles:
        page_py = wiki.page(title)
        if page_py.exists():
            url=page_py.fullurl
            docs=page_py.text
        else:  
            continue
        
        texts = text_splitter.create_documents([docs])
        splits = text_splitter.split_documents(texts)
        all_docs.extend(texts)
    if all_docs==[]:
      return False
    vectorstore = Chroma.from_documents(documents=all_docs, embedding=embeddings_model)
    return vectorstore

def clean_text(text):
    text = '\n'.join([line for line in text.split('\n') if len(line.split()) > 4])
    
    text = re.sub(r'\n{3,3}', '\n', text).strip()
    
    return text

def rag_context(question):
    vectorstore=get_vectorstore(question)
    if vectorstore== False:
      return ""
      
    retriever = vectorstore.as_retriever(K=5,fetch_k=3)
    retrieved_docs=retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return clean_text(formatted_context)
  


#print(rag_context("who does the voice of nala in the lion king?"))