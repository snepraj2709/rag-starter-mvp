import os
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

warnings.filterwarnings('ignore')
load_dotenv()
chat_history =[]

# get the data from pinecode to access it to answer user quzeries
if __name__=='__main__':
    embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], 
        embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
    )    

    res = qa.invoke({"question":"How is new Technologies being used at SEBI.",
                     "chat_history":chat_history})
    print(res)

    history =(res["question"], res["answer"])
    chat_history.append(history)

    res = qa.invoke({"question":"Continuining with the last question, Is Gen AI a technology SEBI used this year",
                     "chat_history":chat_history})
    print(res)