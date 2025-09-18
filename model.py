from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import logging
import os
import re

logger = logging.getLogger(__name__)

load_dotenv()

MAX_INPUT_LENGTH = 500


def sanitize_input(text: str) -> str:
    """
    Basic sanitization for user medical queries.
    - Strips leading/trailing whitespace
    - Enforces maximum length
    - Removes common prompt injection patterns
    """
    text = text.strip()

    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(f"Input too long ({len(text)} chars). Please keep questions under {MAX_INPUT_LENGTH} characters.")

    injection_patterns = [
        r"ignore ((all|previous|prior)\s+)*(instructions?|rules?|prompts?)",
        r"you are now",
        r"disregard (all |your )?(previous |prior )?(instructions?|rules?)",
        r"new instructions?:",
        r"system:",
        r"<\|.*?\|>",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValueError("Input contains disallowed content.")

    return text


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "medical_qa_bge_large_en")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None


rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a reliable and empathetic AI medical assistant trained to answer user questions using only the retrieved information provided below.

If any document clearly answers the question, use that document only.
If the documents contradict each other, are unclear, incomplete, or not relevant, respond with:
"I recommend speaking to a medical professional for accurate advice."

Your response must:
- Be based strictly on the context snippets retrieved.
- Avoid adding any external knowledge or assumptions.
- Be concise, medically responsible, and human-centered.
- Use clear and professional language appropriate for a general audience.
- NEVER provide medication dosages, diagnoses, or treatment plans.
- Do not infer, guess, or summarize beyond the exact information provided.
- Respond with a single line or short paragraph as your FINAL ANSWER only.

CONTEXT SNIPPETS (from verified medical sources):
{context}
""",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def generate_safe_answer(user_question: str, retriever, llm, chat_history=None, langfuse_handler=None):
    if chat_history is None:
        chat_history = []

    try:
        user_question = sanitize_input(user_question)
    except ValueError as e:
        return f"I couldn't process that input: {e}"

    prefixed_query = f"Represent this sentence for searching relevant passages: {user_question}"
    try:
        docs = retriever.invoke(prefixed_query)
    except Exception as e:
        logger.error("Retriever invocation failed: %s", e)
        return (
            "I'm currently unable to search my knowledge base. "
            "Please try again later or consult a medical professional."
        )

    if not docs:
        return "I'm sorry, I couldn't find relevant information. Please consult a medical professional."

    context_snippets = "\n\n".join([
        f"- Q: {doc.page_content}\n  A: {doc.metadata.get('answer', '')}" for doc in docs
    ])
    chain = rag_prompt | llm
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    try:
        return chain.invoke({
            "context": context_snippets,
            "question": user_question,
            "chat_history": chat_history,
        }, config=config)
    except Exception as e:
        logger.error("LLM invocation failed: %s", e)
        return "I'm currently unable to process your request. Please try again or consult a medical professional."
