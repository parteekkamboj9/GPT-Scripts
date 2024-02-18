from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.llms.openai import OpenAI
from langchain.vectorstores.pinecone import Pinecone
import os
import pinecone

# replace the values before run
PINECONE_API_KEY = "enter-your-pineone-api-key-here"
PINECONE_ENV = "enter-your-pineone-api-env-here"
os.environ['OPENAI_API_KEY'] = "enter-your-openai-api-key-here"
index_name = "enter-your-pineone-inede-name-here"
namespace = "enter-your-index-namespace-here"

# write your query
query = 'what is the key points of context'

# pinecone initialization
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
)

# define LLM model (Ollama)
llm = Ollama(model="mistral", verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
# llm = OpenAI()

# creating pinecone object
docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=OpenAIEmbeddings(), namespace=namespace)

# create an object of RetrievalQA of Langchain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
)

# get answer from Retrieval
response = qa({"query": query})
print("Answer:", response['result'])
