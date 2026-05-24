import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_core.embeddings import Embeddings
from typing import List





# ------------------- pinecone config 



from Embedding import MyHuggingFaceEmbeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv




# ------------------- embedding with huggingface -------------------



load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "medical-index"
index = pc.Index(index_name)
embeddings = MyHuggingFaceEmbeddings(index=index)


# ----------  file  ----------------------

# r means read only mode
# input your document in text.txt
data_file = Path(__file__).resolve().parent / "train_pure_text" / "medical_txt"
with data_file.open("r", encoding="utf-8") as f:
    document = f.read()


# ---------------- split text into chunks ----------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，"] #移除换行符，句号，感叹号，问号，逗号等分隔符
)

chunks = text_splitter.split_text(document)


# ------------  store embbeding and text as vector in pinecone


# Add chunks to vector store
embeddings.vector_store.add_texts(chunks)


