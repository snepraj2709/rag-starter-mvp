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
def get_response(user_query: str, chat_history=None):
    """
    This RAG chain returns res.json() if the returned object has that method,
    otherwise returns the object itself (commonly a dict).
    """
    if chat_history is None:
        chat_history = []

    embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    res = qa.invoke({"question": user_query, "chat_history": chat_history})

    try:
        return res.json()
    except AttributeError:
        return res


if __name__ == "__main__":
    # allow running directly: take CLI args or prompt input
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your question: ").strip()

    result = get_response(query)
    print(result)