import openai
import os
from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0, openai_api_key=openai.api_key)

text_splitter = CharacterTextSplitter()

with open("space.json", "r", encoding="utf-8") as f:
    space = f.read().encode("ascii", "ignore").decode("ascii")

with open("space_ascii.txt", "w") as f:
    f.write(space)

texts = text_splitter.split_text(space)

print("length of texts: ", len(texts))

docs = [Document(page_content=text) for text in texts]

chain = load_summarize_chain(llm, chain_type="map_reduce")
with get_openai_callback() as cb:
    result = chain.run(docs)
    print(result)
    print("tokens used: ", cb.total_tokens)
