import config
import streamlit as st 
from pymongo import MongoClient
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.chains import RetrievalQA
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate

#configure MongoDB vectorstore and agent
client = MongoClient(config.URI)
db = client["sample_mflix"]
collection = db["movies"]
embedding_model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    credentials_profile_name="default"
)
vectorstore = MongoDBAtlasVectorSearch(
    collection, 
    embedding_model,
    index_name = "vector_index",
    text_key = "plot",
    embedding_key = "embedding",
    metadata_fields=["title", "year", "genres", "plot"] 
    )
# prompt_template =""" 
# You are a movie recommendation system. Use the following pieces of context to answer the question. 
# Please generate answer based on the context only.
# Please do not use any general knowledge.

# Context: {context}
# Question: {question}

# Please provide:
# 1. The movie title (which should be available in the context)
# 2. The movie genre (which should be available in the context)
# 2. A brief plot summary
# Answer:"""

prompt_template =""" 
You are a movie expert. Users asks questions such as movie including movie recommendation and movie story.
Please generate answer based on the context only.
Please do not use any general knowledge.

Context: {context}
Question: {question}

Answer:"""


PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
qa = RetrievalQA.from_chain_type(
    llm = llm,  
    chain_type = "stuff",
    retriever = vectorstore.as_retriever(
        search_kwargs = {"k": 3}
        ),
        chain_type_kwargs = {"prompt": PROMPT},
        return_source_documents = True
    )


#get LLM response
def get_response(input_text):
    temp_res = qa.invoke({"query": input_text})
    res = temp_res["result"]
    return res


#configure streamlit UI
st.markdown("<h1 style='text-align: center;'>MongoDB Atlas Vector Search</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Demo Webpage 🍃</h1>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)


input_text = st.text_area("Input text", label_visibility="collapsed")
go_button = st.button("Go", type="primary")


if go_button:
    response_content = get_response(input_text)
    st.write(response_content)




