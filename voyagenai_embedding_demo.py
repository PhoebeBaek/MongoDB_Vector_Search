import voyageai
import config
import streamlit as st 
from pymongo import MongoClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_aws import ChatBedrock
from langchain.chains import RetrievalQA

vo_api_key = config.VO_API
vo = voyageai.Client(api_key=vo_api_key)

mdb_client = MongoClient(config.URI)
mdb_db = mdb_client["voyage_ai"]
mdb_collection = mdb_db["mdb_event_driven_app"]

# #chunk data
# loader = UnstructuredPDFLoader("/Users/sojeong/study/python/LLM/vector_search/Event-Driven_Apps_with_MongoDB.pdf", mode="single")
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
# docs = text_splitter.split_documents(data)

# #insert embedding data
# MongoDBAtlasVectorSearch.from_documents(
#     documents=docs,
#     embedding=VoyageAIEmbeddings(model="voyage-3-lite", api_key=vo_api_key),
#     collection=mdb_collection,
#     index_name="voyage_index"
# )


vectorstore = MongoDBAtlasVectorSearch(
    mdb_collection, 
    embedding=VoyageAIEmbeddings(model="voyage-3-lite", api_key=vo_api_key),
    index_name = "app_index",
    text_key = "text",
    embedding_key = "embedding"
    )

# configure streamlit UI
st.markdown("<h1 style='text-align: center;'>MongoDB Atlas Vector Search</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Demo Webpage 🍃</h1>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)


input_text = st.text_area("Input text", label_visibility="collapsed")
general_button = st.button("General", type="primary")
rag_button = st.button("RAG", type="primary")

#define llm
llm = ChatBedrock(model_id="amazon.titan-text-express-v1")

#get general answer
def general_response(input_text): 
    llm = ChatBedrock(model_id="amazon.titan-text-express-v1")
    messages = [
        (
            "system",
            "You are a helpful assistant who is an IT expert.",
        ),
        ("human", input_text),
    ]
    ai_msg = llm.invoke(messages)
    return ai_msg.content

if general_button:
    general_res = general_response(input_text)
    st.write(general_res)


# #get RAG response
qa = RetrievalQA.from_chain_type(
    llm = llm,  
    chain_type = "stuff",
    retriever = vectorstore.as_retriever(
        search_kwargs = {"k": 1}
        ),
        return_source_documents = True
    )

def rag_response(input_text):
    temp_res = qa.invoke({"query": input_text})
    source = temp_res["source_documents"][0]
    response = temp_res['result']
    return source, response



if rag_button:
    source, response = rag_response(input_text)
    st.write(response)
    st.write(source)