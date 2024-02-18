import glob
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os
from langchain.document_loaders import UnstructuredFileLoader
import pinecone
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import DeepLake
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Chroma
from langchain.chains import VectorDBQAWithSourcesChain, RetrievalQA

os.environ["OPENAI_API_KEY"] = "<your_openai_key>"

loader = DirectoryLoader('./txt_data/', glob="./*.txt", loader_cls=TextLoader)
data = loader.load()
data[0].metadata["company"]="Back Support AC"
data[1].metadata["company"]="Back Support HD"
data[2].metadata["company"]="Flex CD"
data[3].metadata["company"]="Flex GT"
data[4].metadata["company"]="Flex Heat"
data[5].metadata["company"]="Flex MLT"
data[6].metadata["company"]="Flex NP"
data[7].metadata["company"]="Flex SC"
data[8].metadata["company"]="Knee & Ankle CR"

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings()

pinecone.init(
    api_key= 'your-pinecone-key',
    environment= 'your-pinecone-env'
)
index_name = 'your-pinecone-index'
metadatas = []
for text in texts:
    metadatas.append({
        "company": text.metadata["company"]
    })

Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name, metadatas=metadatas)


docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
# print(docsearch)

llm = ChatOpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")

query = "what are the CASE STUDIES"
docs = docsearch.similarity_search(query, top_k=1, metadata={"company": ["Flex MLT"]})
print(docs[0].page_content)

res = chain.run(input_documents=docs, question=query)
print(res)






################################################################################################

db = Chroma.from_documents(docs, embeddings)

query = "REFERENCES of fiftheen.txt  "
docs = db.similarity_search(query)
print(docs[0].page_content)

db = DeepLake(dataset_path="./my_deeplake/", embedding_function=embeddings)
db.add_documents(docs)
query = "DOSAGE of five.txt file"
docs = db.similarity_search(query)
print(docs[0].page_content)

db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")
db = FAISS.load_local("faiss_index", embeddings)
results_with_scores = db.similarity_search_with_score("what are the INGREDIENTS five.txt ",k=1)
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
    results_list = list(results_with_scores)
    # print(results_list)
    sorted_results = sorted(results_list, key=lambda x: x[1], reverse=True)
    # print(sorted_results)
    highest_result = sorted_results[0][0].page_content
    print(f"Highest Result: {highest_result}")

    highest_score = sorted_results[0][1]
    highest_result = sorted_results[0][0].page_content
    print(f"Highest Score: {highest_score}")
    print(f"Highest Result: {highest_result}")

    sorted_results = sorted(results_with_scores, key=lambda x: x.score, reverse=True)
    print(sorted_results)
    highest_score = sorted_results[0].score
    highest_result = sorted_results[0].page_content



db = FAISS.from_documents(docs, embeddings)
question = "What is the formula for treating acute"
docs = db.similarity_search(question,metadata=True)
# print(docs[0].page_content)
docs_and_scores = db.similarity_search_with_score(question)
# print(docs_and_scores[0])
embedding_vector = embeddings.embed_query(question)
docs_and_scores = db.similarity_search_by_vector(embedding_vector)
db.save_local("faiss_index")
new_db = FAISS.load_local("faiss_index", embeddings)
docs = new_db.similarity_search(question,metadata=True)
# print(docs[0].metadata)

chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0,max_tokens=1500, model_name='text-davinci-003'),
                                                vectorstore=new_db)

result = chain({"question": str(question)})
print(result)




################################################################################



file_paths = [
    "5.103 Back Support (AC) [Back Support (Acute)].txt",
    "5.104 Back Support (CR) [Back Support (Chronic)].txt",
    "5.105 Back Support (HD).txt",
    "5.136 Flex (CD).txt",
    "5.137 Flex (GT).txt",
    "5.138 Flex (Heat).txt",
    "5.139 Flex (MLT).txt",
    "5.140 Flex (NP).txt",
    "5.141 Flex (SC).txt",
    "5.142FlexSPR.txt",
    "5.143 Flex (TMX) [Traumanex].txt",
    "5.168 Knee & Ankle (AC) [Knee & Ankle (Acute)].txt",
    "5.169 Knee & Ankle (CR) [Knee & Ankle (Chronic)].txt",
]


from langchain.document_loaders import UnstructuredAPIFileLoader
filenames = ["5.103 Back Support (AC) [Back Support (Acute)].txt", "5.104 Back Support (CR) [Back Support (Chronic)].txt"]
loader = UnstructuredAPIFileLoader(file_path=filenames,)
file_system_paths = []

for file_path in file_paths:
    file_system_path = os.fspath(file_path)
    file_system_paths.append(file_system_path)

loader = UnstructuredFileLoader(file_system_paths)


# # loader = TextLoader(file_paths)
#
# documents = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

query = "What is INGREDIENTS"
docs = db.similarity_search(query)
print(docs[0].page_content)


######################################################################################################

######################################################################################################

texts = text_splitter.create_documents([content])
# print(texts)
# print(texts[0])
metadatas = [
    {"document": '5.103 Back Support (AC) [Back Support (Acute)].txt'},
    {"document": '5.104 Back Support (CR) [Back Support (Chronic)].txt'},
    {"document": '5.105 Back Support (HD).txt'},
    {"document": '5.137 Flex (CD).txt'},
    {"document": '5.137 Flex (GT).txt'},
    {"document": '5.138 Flex (Heat).txt'},


]

documents = text_splitter.create_documents([content,content,content,content,content,content], metadatas=metadatas)
# print(documents)
#
embeddings = OpenAIEmbeddings()


db = FAISS.from_documents(texts, embeddings)

db.save_local("faiss_index")
new_db = FAISS.load_local("faiss_index", embeddings)
query="what is INGREDIENTS?"
docs = new_db.similarity_search(query)
print(docs[0].page_content)

docs_and_scores = db.similarity_search_with_score(query)
print(docs_and_scores[0],"-------------------------")


########################################################################################################


retriever = db.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)
llm = OpenAI(temperature=0, max_tokens=500, model_name='text-davinci-003')
_DEFAULT_TEMPLATE = """
       # You are an helpful assistant where you have to give a answer from the giving text file\n
       give a answer from exact as written in text file \n
        give answer from metadata attribute\n

     Relevant pieces of previous conversation:
     {history}

     Current conversation:
     User: {input}
     Buffy:
     """

PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)
conversation_with_summary = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    memory=memory,
    verbose=True
)
result = conversation_with_summary.predict(input="5.105 Back Support (HD) for  COMPARATIVE ANALYSIS")
print(result)


memory = VectorStoreRetrieverMemory(retriever=retriever)
llm = OpenAI(temperature=1, max_tokens=500, model_name='text-davinci-003')
chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0, max_tokens=1500, model_name='text-davinci-003'), vectorstore=db)
question="what is INGREDIENTS "
result = chain({"question":str(question) })
print(result['answer'])


db.save_local("faiss_index")
new_db = FAISS.load_local("faiss_index", embeddings)
# query = "what aINGREDIENTS of 5.105 Back Support (HD) "
# docs = new_db.similarity_search(query)
# print(docs[0].page_content)
chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0, max_tokens=1500, model_name='text-davinci-003'), vectorstore=new_db)
question="what is INGREDIENTS of 5.103 Back support (HD)"
result = chain({"question":str(question) })
print(result['answer'])


######################################################################################################


retriever = new_db.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)
llm = OpenAI(temperature=0, max_tokens=500, model_name='text-davinci-003')
_DEFAULT_TEMPLATE = """
       You are an helpful assistant that have all the information from the given text data\n
       Please give a full answer/n


     Relevant pieces of previous conversation:
     {history}

     (You do not need to use these pieces of information if not relevant)

     Current conversation:
     User: {input}
     Buffy:
     """

PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)
conversation_with_summary = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    memory=memory,
    verbose=True
)
result = conversation_with_summary.predict(input="what is CAUTIONS & CONTRAINDICATIONS refers in document 1")
print(result)

# query = "what are the INGREDIENTS"
# docs = db.similarity_search(query)

print(docs[0].page_content)
print(docs)
