# model.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import PromptTemplate
from langchain_qdrant import Qdrant
import textwrap
try:
    from langfuse.langchain import CallbackHandler
except Exception:
    CallbackHandler = None

# Configuration (vectorstore and retriever are passed from streamlit_app.py)


# 1. Connect to Qdrant

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"
QDRANT_COLLECTION_NAME = "medical_qa_bge_large_en"
QDRANT_URL = "http://localhost:6333"   
QDRANT_API_KEY = None 

# 4. Prompt Template
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=textwrap.dedent("""
You are a reliable and empathetic AI medical assistant trained to answer user questions using only the retrieved information provided below.
If any document clearly answers the question, use that document only.
If the documents contradict each other, are unclear, incomplete, or not relevant, respond with:

"I recommend speaking to a medical professional for accurate advice."


Your response must:

-Be based strictly on the context snippets retrieved.
-Avoid adding any external knowledge or assumptions.
-Be concise, medically responsible, and human-centered.
-Use clear and professional language appropriate for a general audience.

QUESTION:
{question}

CONTEXT SNIPPETS (from verified medical sources):
{context}
---

INSTRUCTIONS TO THE MODEL:
- NEVER start your response with "According to the provided context snippets" or similar phrases.
- Use the retrieved context only if it it clearly answers the question.
- If the context contains conflicting information, respond:
    "I recommend speaking to a medical professional for accurate advice."
- You may merge information from multiple sources ONLY IF they are consistent.
- Do not infer, guess, or summarize beyond the exact information provided.
- Do not merge partial answers from multiple sources unless they agree fully.
- Do not provide medication dosages, diagnoses, or treatment plans.
- Do not apologize, explain limitations, or restate the context or question.
- Respond with a single line or short paragraph that represents your FINAL ANSWER only.

FINAL ANSWER:
""")
)


# Generate Answer
def generate_safe_answer(user_question: str, retriever, llm, langfuse_handler=None):
    prefixed_query = f"Represent this sentence for searching relevant passages: {user_question}"
    docs = retriever.invoke(prefixed_query)

    if not docs:
        return "I'm sorry, I couldn't find relevant information. Please consult a medical professional."
    print("\n📥 Retrieved Documents:")
    for i, doc in enumerate(docs):
        print(f"--- Document {i+1} ---")
        print(f"Question: {doc.page_content}")
        print(f"Answer: {doc.metadata.get('answer')}")
        print(f"Tags: {doc.metadata.get('tags')}\n")
    context_snippets = "\n\n".join([
        f"- Q: {doc.page_content}\n  A: {doc.metadata.get('answer', '')}" for doc in docs
    ])
    prompt = rag_prompt.format(context=context_snippets, question=user_question)
    print("\n📄 Final Context Passed to Model:\n", context_snippets)
    print("\n🧠 Final Prompt:\n", prompt)
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    return llm.invoke(prompt, config=config)