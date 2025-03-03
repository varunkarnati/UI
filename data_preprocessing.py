import pdfplumber
import uuid
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone as pineC, ServerlessSpec
from langchain_pinecone import Pinecone
import os 
from dotenv import load_dotenv
load_dotenv()

def extract_pdf(file_path):
    texts=[]
    tables=[]
    # Open the PDF and extract pages
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text())
            # print(text) # Extract plain text
            if page.extract_tables():
                tables.append(page.extract_tables())
                # Extract tables
    return texts, tables
def summarize_data(texts,tables):
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text that perfectly describes the table in starting 2 sentences.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    #
    # Summary chain
    model = ChatGroq(temperature=0, model="llama-3.1-8b-instant",api_key=os.environ["GROQ_API_KEY"])
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    # Summarize extracted text
    text_summaries = []
    if texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})

    # Summarize extracted tables
    tables_html = [str(table) for table in tables]  # Convert tables to string format
    table_summaries = []
    if tables_html:
        table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 5})
    return texts,text_summaries,tables,table_summaries

def create_vectorstore():

    model_name = "intfloat/multilingual-e5-large-instruct"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # index= pc.Index("gaido-rag")
    # The vectorstore to use to index the child chunks
    # vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=hf)

    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"
    
    pc = pineC(api_key=os.environ["PINECONE_API_KEY"])

    index_name = "gaidorag"
    text_field = "text"
    cloud ='aws'
    region = 'us-east-1'

    spec = ServerlessSpec(cloud=cloud, region=region)
    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pc.list_indexes().names():
        # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=1024,  # dimensionality of text-embedding-ada-002
            metric='cosine',
            spec=spec
        )
    # switch back to normal index for langchain
    index = pc.Index(index_name)

    vectorstore = Pinecone(
        index, hf, text_field
    )


    
# The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    return retriever
def embed_docs(retriever,texts,text_summaries,tables,table_summaries):
    # Add texts
    id_key = "doc_id"
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))


def process_docs(file_path):
    texts,tables=extract_pdf(file_path)
    texts,text_summaries,tables,table_summaries=summarize_data(texts,tables)
    retriever=create_vectorstore()
    embed_docs(retriever,texts,text_summaries,tables,table_summaries)
    return retriever
