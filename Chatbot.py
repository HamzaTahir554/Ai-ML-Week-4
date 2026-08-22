from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "document.pdf"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# This value will be improved later
TOP_K = 5

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OLLAMA_MODEL = "llama3.2"


# ============================================================
# 1. LOAD PDF
# ============================================================

print("Loading PDF...")

reader = PdfReader(PDF_PATH)

pages = []

for page_number, page in enumerate(
    reader.pages,
    start=1
):

    text = page.extract_text()

    if text:

        pages.append({
            "page": page_number,
            "text": text
        })


print(
    f"Pages loaded: {len(pages)}"
)


# ============================================================
# 2. CREATE CHUNKS
# ============================================================

print("Creating chunks...")

chunks = []

for page in pages:

    text = page["text"]

    page_number = page["page"]

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end]

        chunks.append({
            "text": chunk,
            "page": page_number
        })

        start += (
            CHUNK_SIZE -
            CHUNK_OVERLAP
        )


print(
    f"Total chunks: {len(chunks)}"
)


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


texts = [
    chunk["text"]
    for chunk in chunks
]


print("Creating embeddings...")

embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)


# ============================================================
# 4. CREATE CHROMA DATABASE
# ============================================================

print("Creating Chroma database...")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="document_collection"
)


# ============================================================
# 5. STORE DOCUMENTS
# ============================================================

print("Storing vectors...")

collection.add(

    ids=[
        f"chunk_{i}"
        for i in range(len(chunks))
    ],

    documents=texts,

    embeddings=embeddings.tolist(),

    metadatas=[
        {
            "page": chunk["page"]
        }

        for chunk in chunks
    ]
)


print("RAG database ready.")


# ============================================================
# TOOL 1 — RAG SEARCH
# ============================================================

def rag_search(question):

    print("\n[RAG TOOL CALLED]")


    # Convert question to embedding

    question_embedding = (
        embedding_model.encode(
            [question]
        )[0]
    )


    # Search Chroma

    results = collection.query(

        query_embeddings=[
            question_embedding.tolist()
        ],

        n_results=TOP_K
    )


    documents = results[
        "documents"
    ][0]

    metadata = results[
        "metadatas"
    ][0]


    # Build context

    context_parts = []

    for i, document in enumerate(
        documents
    ):

        page = metadata[i]["page"]

        context_parts.append(
            f"[Page {page}]\n{document}"
        )


    context = "\n\n".join(
        context_parts
    )


    return context


# ============================================================
# TOOL 2 — CALCULATOR
# ============================================================

def calculator(expression):

    print("\n[CALCULATOR TOOL CALLED]")


    try:

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            {}
        )

        return str(result)

    except Exception:

        return "Invalid calculation."


# ============================================================
# AGENT STATE
# ============================================================

class AgentState(TypedDict):

    question: str

    context: str

    answer: str


# ============================================================
# AGENT NODE
# ============================================================

def agent_node(state):

    question = state["question"]


    prompt = f"""
You are an intelligent assistant.

You have access to two tools:

1. RAG SEARCH
   Use this when the user asks about
   information contained in the document.

2. CALCULATOR
   Use this when the user asks for
   mathematical calculations.

Question:

{question}

Decide what type of question this is.

If it is a document question,
respond with:

RAG

If it is a mathematical question,
respond with:

CALCULATOR

Otherwise respond with:

NONE
"""


    response = ollama.chat(

        model=OLLAMA_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    decision = (
        response["message"]["content"]
        .strip()
        .upper()
    )


    if "CALCULATOR" in decision:

        result = calculator(question)

        return {
            "question": question,
            "context": "",
            "answer": result
        }


    elif "RAG" in decision:

        context = rag_search(
            question
        )


        return {
            "question": question,
            "context": context,
            "answer": ""
        }


    else:

        return {
            "question": question,
            "context": "",
            "answer":
                "I can answer questions "
                "about the document or "
                "perform calculations."
        }


# ============================================================
# ANSWER NODE
# ============================================================

def answer_node(state):

    question = state["question"]

    context = state["context"]


    prompt = f"""
Answer the question using ONLY the
provided document context.

If the answer is not available in
the context, say:

"I don't know based on the provided document."

Do not invent information.

Always mention the page number when
using information from the document.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""


    response = ollama.chat(

        model=OLLAMA_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    answer = response[
        "message"
    ]["content"]


    return {
        "question": question,
        "context": context,
        "answer": answer
    }


# ============================================================
# LANGGRAPH
# ============================================================

graph = StateGraph(
    AgentState
)


graph.add_node(
    "agent",
    agent_node
)

graph.add_node(
    "answer",
    answer_node
)


graph.add_edge(
    START,
    "agent"
)

graph.add_edge(
    "agent",
    "answer"
)

graph.add_edge(
    "answer",
    END
)


app = graph.compile()


# ============================================================
# CHATBOT
# ============================================================

print("\n====================================")
print("      RAG AGENT CHATBOT")
print("====================================")

print(
    "Ask questions about your PDF."
)

print(
    "You can also ask calculations."
)

print(
    "Type 'exit' to stop.\n"
)


while True:

    question = input(
        "You: "
    )


    if question.lower() == "exit":

        print("Goodbye!")

        break


    result = app.invoke({

        "question": question,

        "context": "",

        "answer": ""

    })


    print("\nBot:")

    print(
        result["answer"]
    )

    print("\n" + "-" * 50)