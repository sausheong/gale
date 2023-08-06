import os
import argparse
import pinecone
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

print("\033[96mPinecone management tool\033[0m")
# upload a file or delete all the files
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", required=False, help="file path of document to upload")
parser.add_argument("-d", "--delete_all", required=False, help="if true delete all vectors in index")
args = vars(parser.parse_args())

# set up Pinecone
print("\033[93m- Setting up Pinecone settings from .env file\033[0m")
embeddings = OpenAIEmbeddings()
index_name = os.getenv("PINECODE_INDEX")
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  
    environment=os.getenv("PINECONE_ENV"),  
)

print("\033[93m- Checking if index exists\033[0m")
# First, check if our index already exists. If it doesn't, we create it
if index_name not in pinecone.list_indexes():
    # we create a new index
    print(f"\033[93m- Creating index: {index_name}\033[0m")
    pinecone.create_index(
      name=index_name,
      metric='cosine',
      dimension=1536  
    )
else:
    print(f"\033[93m- Index exists: {index_name}\033[0m")

print("\033[93m- Creating vectordb from index\033[0m")
index = pinecone.Index(index_name)
vectordb = Pinecone(index, embeddings.embed_query, "text")

if args["delete_all"]:
    print("\033[93m- Delete all from vectordb\033[0m")
    vectordb.delete(delete_all=True)

elif args["load"]:
    print(f"\033[93m- Loading file {args['load']} as document\033[0m")   
    loader = UnstructuredFileLoader(file_path=args['load'])
    documents = loader.load()
    
    print("\033[93m- Splitting up document into text\033[0m")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    print("\033[93m- Adding document into vector database\033[0m")
    ids = vectordb.add_documents(documents=texts)

print("\033[96mEnd\033[0m")