from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore # LangChain'in Qdrant entegrasyonu

loader = TextLoader("sample_data.txt")  #langchain textloader used for .txt files
documents = loader.load()        #loaded to the memory

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

load_dotenv() #loading env file and getting api key variable

api_key = os.getenv("OPENAI_API_KEY")


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

qdrant = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="test_collection"
)
print("Tebrikler! Veriler başarıyla parçalandı ve Qdrant'a yüklendi.")