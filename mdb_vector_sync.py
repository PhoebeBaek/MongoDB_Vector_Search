#기존 데이터에 vector data를 추가해서 semantic search 진행
import config
from pymongo import MongoClient
from langchain_aws import BedrockEmbeddings
from langchain_aws import BedrockLLM
from langchain_community.vectorstores import MongoDBAtlasVectorSearch

client = MongoClient(config.URI)
db = client['sample_mflix']
collection = db['movies']

embedding_model = BedrockEmbeddings(
    model_id='amazon.titan-embed-text-v2:0',
    credentials_profile_name='default'
    )


def get_data():
    temp_data = collection.find({})
    movie_data = list(temp_data)
    return movie_data

def upsert_embedding():
    movie_data = get_data()
    total = len(movie_data)
    print(f"Processing {total} documents...")

    for i, data in enumerate(movie_data, 1):
        try:
            if 'plot' in data and data['plot']:
                embedding=embedding_model.embed_query(data['plot'])
                collection.update_one(
                    {'_id': data['_id']},
                    {'$set': {'embedding': embedding}},
                    upsert=True
                )
                if i % 100==0:
                    print(f'Processed {i}/{total} documents...')
        except Exception as e:
            print(f"Error processing document {data['_id']}: {e}")
    print("Finished processing all documents!")


upsert_embedding()
print("Embedding data insertion complete!")


