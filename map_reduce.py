from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain import OpenAI
from dotenv import load_dotenv
import tiktoken
import openai
import os


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

chain = load_qa_chain(
    OpenAI(temperature=0), chain_type="map_reduce", return_map_steps=True
)

while True:
    with get_openai_callback() as cb:
        query = input("query: ")
        docs = db.similarity_search(query, k=5)
        result = chain(
            {"input_documents": docs, "question": query}, return_only_outputs=True
        )
        print(result["output_text"])
        print("tokens:", cb.total_tokens)
