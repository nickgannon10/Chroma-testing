from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

persist_directory = "db"
embeddings = OpenAIEmbeddings()

if not os.path.exists(persist_directory):
    with open("book.txt", "r", encoding="utf-8") as f:
        book = f.read().encode("ascii", "ignore").decode("ascii")
    with open("book_ascii.txt", "w") as f:
        f.write(book)

    print("loading book")

    loader = TextLoader("book_ascii.txt")

    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("embedding book")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
else:
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

prompt_template = """Use the following pieces of context to answer the question as the end by summarizing the context. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.5),
    chain_type="stuff",
    retriever=db.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
    return_source_documents=True,
)

while True:
    query = input("question: ")
    result = qa(query)
    print(result["result"])
    print(result["source_documents"])
