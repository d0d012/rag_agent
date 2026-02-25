from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
load_dotenv() #loading env file and getting api key variable

api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="test_collection"
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
)

messages = [SystemMessage (content="Sen bir yapay zeka asistanısın. Kullanıcının sorusunu vektör veritabanında aramak için 3 farklı alternatif arama sorgusu (query) üret. Sadece sorguları alt alta yaz."),
            HumanMessage(content="Go dilindeki arayüzler, diğer dillerdeki regresyon testlerini nasıl etkiler?")]

response = llm.invoke(messages)
lines = response.content.splitlines()
lines.append("Go dilindeki arayüzler, diğer dillerdeki regresyon testlerini nasıl etkiler?")

all_results = []

for query in lines:
    all_results.append(qdrant.similarity_search(k=3, query=query)) #list of lists (list of related documents to each query)

rrf_scores = {}

for result in all_results:
    for i, document in enumerate(result):
        score = float(1 / (i+61))
        if document.page_content in rrf_scores:
            rrf_scores[document.page_content] += score
        else:
            rrf_scores.setdefault(document.page_content,score)

sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

print("\n--- Best Results ---")
for j in range(min(3, len(sorted_docs))):
    metin = sorted_docs[j][0]
    score = sorted_docs[j][1]
    print(f"\n[Rank {j+1}] RRF Score: {score:.4f}")
    print(f"Text: {metin[:200]}...") 

# --- ÜST KISIM AYNI ---

# Soruyu bir değişkene atıyoruz ki aşağıda da kullanabilelim
original_query = "Go dilindeki arayüzler, diğer dillerdeki regresyon testlerini nasıl etkiler?"

messages = [
    SystemMessage(content="Sen bir yapay zeka asistanısın. Kullanıcının sorusunu vektör veritabanında aramak için 3 farklı alternatif arama sorgusu (query) üret. Sadece sorguları alt alta yaz."),
    HumanMessage(content=original_query) # Değişkeni burada kullandık
]

response = llm.invoke(messages)
lines = response.content.splitlines()
lines.append(original_query) # Orijinal soruyu da varyasyonlara ekledik

all_results = []

for query in lines:
    all_results.append(qdrant.similarity_search(k=3, query=query))

rrf_scores = {}

for result in all_results:
    for i, document in enumerate(result):
        score = float(1 / (i+61))
        if document.page_content in rrf_scores:
            rrf_scores[document.page_content] += score
        else:
            rrf_scores.setdefault(document.page_content, score)

sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

print("\n--- Best Results ---")
for j in range(min(3, len(sorted_docs))):
    metin = sorted_docs[j][0]
    score = sorted_docs[j][1]
    print(f"\n[Rank {j+1}] RRF Score: {score:.4f}")
    print(f"Text: {metin[:200]}...") 

print("\n\n--- GENERATION ---")

context_text = "\n\n---\n\n".join([doc[0] for doc in sorted_docs[:2]])

generation_messages = [ #prompt scheme
    SystemMessage(content=(
        "Sen uzman bir yazılım mimarısın. "
        "Kullanıcının sorusunu SADECE sana verilen 'Bağlam' metnini kullanarak cevapla. "
        "Eğer sorunun cevabı bağlamda açıkça belirtilmiyorsa veya iki kavram arasında bağlamda bir ilişki kurulmamışsa, "
        "uydurma ve dürüstçe 'Verilen dokümanlarda bu konuyla ilgili bir bağlantı/cevap bulunmamaktadır.' de."
    )),
    HumanMessage(content=f"Bağlam:\n{context_text}\n\nKullanıcının Sorusu: {original_query}")
]

#llm invoke
print("LLM sağlanan bağlamı okuyor ve cevabı üretiyor...\n")
final_response = llm.invoke(generation_messages)

print("🤖 AI YANITI:")
print(final_response.content)