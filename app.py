import tiktoken
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def read_file(filename):
  with open(filename, 'r') as file:
    return file.read()
  

def main():
  load_dotenv()
  long_text = read_file("constitution.txt")

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, 
                                                 chunk_overlap=50, 
                                                 separators=['\n\n', '\n', '.', ' ',''])
  chunks = text_splitter.split_text(long_text)

  embeddings = OpenAIEmbeddings()
  vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
  query = "What is the prime duty of the government?"
  retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4})
  qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

  response = qa(query)
  print(response)

  
if __name__ == "__main__":
    main()