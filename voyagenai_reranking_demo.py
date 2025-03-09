import voyageai
import config
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader

vo_api_key = config.VO_API
vo = voyageai.Client(api_key=vo_api_key)

mdb_client = MongoClient(config.URI)
mdb_db = mdb_client["voyage_ai"]
mdb_collection = mdb_db["mdb_event_driven_app"]

#connect to vector store
vectorstore = MongoDBAtlasVectorSearch(
    mdb_collection, 
    embedding=VoyageAIEmbeddings(model="voyage-3-lite", api_key=vo_api_key),
    index_name = "app_index",
    text_key = "text",
    embedding_key = "embedding"
    )

query = "How is Atlas Triggers charged?"

results = vectorstore.similarity_search_with_score(
    query=query,
    k=3
)

print("=============================Initial search===================================")
for document, score in results:
    parsed_data = [{'page_content': document.page_content, 'score': score}]
    for item in parsed_data:
        print(item)
    print("==============================================================================")    


# Output the parsed data
for page_content, score in parsed_data:
    print(f"Score: {score}\nPage Content: {page_content}\n")


docs = [result[0] for result in results]
doc_contents = [doc.page_content for doc in docs]

#rerank
reranking = vo.rerank(query, documents=doc_contents, model="rerank-2-lite", top_k=3)
print("====================================Rerank result===================================")
for r in reranking.results:
    parsed_data = [{'page_content': r.document, 'score': r.relevance_score}]
    print(parsed_data)
    print("================================================================================")