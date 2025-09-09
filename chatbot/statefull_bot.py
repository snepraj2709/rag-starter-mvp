import os
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.schema import HumanMessage, AIMessage

warnings.filterwarnings('ignore')
load_dotenv()

def _normalize_chat_history(chat_history):
    """
    Accepts:
      - None
      - list of tuples [(human, ai), ...]
      - list of dicts [{"role":..., "content":...}, ...]
      - list of BaseMessage objects
    Returns list of tuples [(human, ai), ...] which ConversationalRetrievalChain accepts.
    """
    if not chat_history:
        return []

    # If list of tuples already, return as-is
    if all(isinstance(x, tuple) and len(x) == 2 for x in chat_history):
        return chat_history

    # If list of BaseMessage objects
    if all(hasattr(x, "type") or hasattr(x, "role") for x in chat_history):
        # convert to tuples by pairing user->assistant
        pairs = []
        last_user = None
        for msg in chat_history:
            # try to get role and text in various possible shapes
            role = getattr(msg, "type", None) or getattr(msg, "role", None)
            text = getattr(msg, "content", None) or getattr(msg, "text", None)
            if role == "human" or role == "user":
                last_user = text
            elif role == "ai" or role == "assistant":
                if last_user is not None:
                    pairs.append((last_user, text))
                    last_user = None
        return pairs

    # If list of dicts
    if all(isinstance(x, dict) for x in chat_history):
        pairs = []
        last_user = None
        for m in chat_history:
            role = m.get("role")
            text = m.get("content") or m.get("question") or ""
            if role == "user":
                last_user = text
            elif role == "assistant" and last_user is not None:
                pairs.append((last_user, text))
                last_user = None
        return pairs

    return []

def get_response(user_query: str, chat_history=None):
    if chat_history is None:
        chat_history = []

    chat_history_pairs = _normalize_chat_history(chat_history)
    embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    res = qa.invoke({"question": user_query, "chat_history": chat_history_pairs})

    try:
        return res.json()
    except AttributeError:
        return res
