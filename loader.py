from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
#from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
import argparse
import os
import shutil
from getpass import getpass
from tqdm import tqdm
              
chroma_path = 'nomicDB'
path = 'pdfFiles'

def load_document():
  document_loader = PyPDFDirectoryLoader(path)
  return document_loader.load()

def split_documents(documents: list[Document]):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 80, length_function = len, is_separator_regex = False,)
  return text_splitter.split_documents(documents)

#we want the embedding function for creating the vector db and the query to be same
def get_embedding_function():
  # embeddings = BedrockEmbeddings(
  #     credentials_profile_name = "default", region_name = "us-east-1"
  # )
  embeddings = OllamaEmbeddings(
      model = "nomic-embed-text"
  )
  return embeddings
#each chunk has a source file path and page number
#we can combine path, page number, and chunk number to get a uniq id for each chunk
#if we want to append in the vector db

def create_chunk_id(chunks):
  last_page_id = None
  current_chunk_index = 0

  for chunk in chunks:
      source = chunk.metadata.get("source")
      page = chunk.metadata.get("page")
      current_page_id = f"{source}:{page}"

      # If the page ID is the same as the last one, increment the index.
      if current_page_id == last_page_id:
          current_chunk_index += 1
      else:
          current_chunk_index = 0

      # Calculate the chunk ID.
      chunk_id = f"{current_page_id}:{current_chunk_index}"
      last_page_id = current_page_id

      # Add it to the page meta-data.
      chunk.metadata["id"] = chunk_id
  return chunks



#using chromadb for our vector database of embeddings
#if want to create a db and append new embeddings we need to tag every chunk with a string id
def add_to_chroma(chunks: list[Document]):
  db = Chroma(
      persist_directory = chroma_path, 
      embedding_function = get_embedding_function()
      )
  new_id_chunks = create_chunk_id(chunks)

  existing_items = db.get(include=[])
  existing_ids = set(existing_items["ids"])
  print("exiting documents:", len(existing_ids))

  new_chunks = [] #only add documents that do not exist in the db already
  for chunk in tqdm(new_id_chunks):
    if chunk.metadata["id"] not in existing_ids: #if not in existing set than  it is new
      new_chunks.append(chunk)
      
  print(len(new_chunks))
  print(new_chunks)

  if len(new_chunks):
    new_ids = [chunk.metadata["id"] for chunk in new_chunks]
    db.add_documents(new_chunks, ids = new_ids)
    db.persist()
    print("New documents added")


def clear_database():
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


def main():
    # Create (or update) the data store.
    documents = load_document()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
#main()
