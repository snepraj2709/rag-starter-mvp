import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# get the data from pinecode to access it to answer user quzeries
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], 
    embedding=embeddings
)

chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

qa = RetrievalQA.from_chain_type(
    llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
)    

res = qa.invoke("Which machine and optimizer was used to train the models.")
print(res) 

res = qa.invoke("Why did we vary the number of attention heads and the attention key and value dimensions")
print(res)

res =qa.invoke('Can you elaborate more on the last question regarding transaction id 1236?')
print(res)