!pip install langchain-community langchain-core
!pip install --upgrade langchain-aws
!pip install sentence_transformers
!pip install pymongo
!pip install langchain_mongodb

import getpass, os, pymongo, pprint
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain.chains import RetrievalQA

#local이면 aws credential도 추가필요
os.environ['mdb_password'] = 'a'

CONNECTION_STRING = f"mongodb+srv://sojeong:{os.environ['mdb_password']}@cluster0.ac173wv.mongodb.net/"
client = MongoClient(CONNECTION_STRING)
dbname=client['Cluster0']
collection_name = dbname["product"]

llm = Bedrock(
    model_id="amazon.titan-text-express-v1"
)
embeddings = BedrockEmbeddings(region_name="us-east-1") 
db_name = "Cluster0"
collection_name = "product"
atlas_collection = client[db_name][collection_name]
vector_search_index = "vector_index"

loader = CSVLoader(file_path="./RandomData.csv")
data = loader.load()


# insert embeddings into MongoDB
vector_search = MongoDBAtlasVectorSearch.from_documents(
    documents = data,
    embedding = embeddings,
    collection = atlas_collection,
    index_name = vector_search_index
)

# create the vector store
vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    CONNECTION_STRING,
    "Cluster0.product",
    embeddings,
    index_name=vector_search_index
)

#create an agent
qa = RetrievalQA.from_chain_type(
    llm = llm, 
    chain_type = "stuff", 
    #retrieve data from vector store
    retriever = vector_search.as_retriever())

#ask questions
res = qa.invoke("Tell me a product with the highest rating.")
res["result"]
