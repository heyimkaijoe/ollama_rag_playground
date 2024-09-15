from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

llm = Ollama(model='mistral')
loader = TextLoader('maxwell_mead.txt')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=5,
    separators=[' ', ',', '\n'],
)
splitted_doc = text_splitter.split_documents(loader.load())
print(splitted_doc)

embeddings = OllamaEmbeddings(model='nomic-embed-text')

vector_db = Chroma.from_documents(
    documents=splitted_doc,
    embedding=embeddings,
    persist_directory='db',
    collection_name='maxwell_mead',
)

retriever = vector_db.as_retriever(search_kwargs={'k': 3})

system_prompt = "If you don't know the answer, please feel free to say 'I don't know'. Here's the context:\n\n{context}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('user', 'Question: {input}'),
    ]
)

document_chain = create_stuff_documents_chain(llm, prompt_template)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

context = []
input_text = input('What do you want to ask?\n>>> ')

while input_text.lower() != 'bye':
    response = retrieval_chain.invoke({ 'input': input_text, 'context': context })
    context = response['context'] # update context

    print(response['answer'])

    input_text = input('>>> ')
