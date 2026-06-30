import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        self.db = None
        self.qa_chain = None

    def load_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents

    def process_documents(self, documents):
        chunks = self.text_splitter.split_documents(documents)

        self.db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="vectorstore"
        )

        self.db.persist()

    def create_qa_chain(self):
        retriever = self.db.as_retriever(search_kwargs={"k": 3})

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True
        )

    def ask(self, question):
        if not self.qa_chain:
            raise Exception("QA chain not initialized")

        return self.qa_chain.invoke({"query": question})
