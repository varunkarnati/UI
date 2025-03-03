from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from base64 import b64decode
import os
from dotenv import load_dotenv
load_dotenv


def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    # Extract text context
    context_text = ""
    if docs_by_type.get("texts"):
        for text_element in docs_by_type["texts"]:
            if isinstance(text_element, list):
                # Flatten nested lists before joining
                flat_text = " ".join(
                    " ".join(map(str, sub_element)) if isinstance(sub_element, list) else str(sub_element)
                    for sub_element in text_element
                )
                context_text += flat_text + "\n"
            else:
                context_text += str(text_element) + "\n"

    # Extract table context
    context_tables = ""
    if docs_by_type.get("tables"):
        for table in docs_by_type["tables"]:
            table_str = "\n".join([" | ".join(map(str, row)) for row in table])  # Convert table rows to strings
            context_tables += f"\nTable:\n{table_str}\n"

    # Construct prompt with context (including tables)
    prompt_template = f"""
    Answer the question based only on the following context only Ground Truth Final Answer, which includes text and tables.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Be specific

    Context:
    {context_text}

    {context_tables}

    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    # If images are provided, include them
    # if docs_by_type.get("images"):
    #     for image in docs_by_type["images"]:
    #         prompt_content.append(
    #             {
    #                 "type": "image_url",
    #                 "image_url": {"url": f"data:image/jpeg;base64,{image}"},
    #             }
    #         )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )

def create_rag_chain(retriever):
    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | ChatGroq(model="llama-3.3-70b-versatile",api_key=os.environ["GROQ_API_KEY"])
        | StrOutputParser()
    )

    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            | ChatGroq(model="llama-3.3-70b-versatile",api_key=os.environ["GROQ_API_KEY"])
            | StrOutputParser()
        )
    )
    return chain, chain_with_sources

def invoke_chain(chain):
    response = chain.invoke(
    "What is the policy start and expiry date?"
    

    )

    return response
