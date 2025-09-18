# Medica

Medica is a Streamlit medical assistant that combines retrieval-augmented medical Q&A, voice interaction, text-to-speech responses, and appointment booking in one workflow.

The project is built to show how a healthcare assistant can move beyond a simple chatbot: users can ask medical questions, speak instead of typing, hear responses back, and book appointments through a connected calendar flow.

Medica is for informational and demonstration use. It does not replace professional medical advice, diagnosis, or treatment.

## Highlights

- Medical Q&A powered by RAG, using Qdrant to retrieve relevant medical knowledge before answering.
- Voice-first interaction with explicit start and stop recording controls.
- Whisper transcription for spoken questions.
- Coqui TTS audio playback for assistant responses.
- Conversation memory for more natural follow-up questions.
- Appointment booking through Radicale/CalDAV.
- Email notification support for booked appointments.
- Environment-based configuration for local or hosted services.
- Optional Langfuse tracing for observability.
- Basic input safety checks before sending questions to the model.

## How It Works

For medical questions, Medica embeds the user query, retrieves relevant context from Qdrant, and sends the context, question, and recent chat history to the LLM through LangChain. This keeps answers grounded in the indexed medical data instead of relying only on the base model.

For voice questions, the user clicks **Ask by Voice** to start recording. Once recording starts, a **Stop Recording** button appears. The recorded audio is transcribed with Whisper, answered through the same RAG pipeline, and the response audio is generated with Coqui TTS.

For appointment booking, Medica uses a separate conversational flow to collect appointment details, check availability through CalDAV, create the appointment, and send an email notification when configured.

## What It Uses

- Streamlit for the UI
- Qdrant for vector search
- LangChain with Ollama `llama3`
- `BAAI/bge-large-en` embeddings by default
- Whisper transcription for voice input
- Coqui TTS for bot audio
- Radicale/CalDAV for appointment booking
- SMTP for appointment email notifications
- Langfuse for optional tracing

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
EMBEDDING_MODEL_NAME=BAAI/bge-large-en
QDRANT_COLLECTION_NAME=medical_qa_bge_large_en

RADICALE_URL=http://localhost:5232/
USERNAME=username
PASSWORD=password

SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=user.email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=user.email@gmail.com

LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3000
```

Start Qdrant:

```bash
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Populate the vector collection if needed:

```bash
python tools/build_vectors/build_qa_vectors-BAAI-bge-large-en.py
```

Make sure Ollama has the model:

```bash
ollama pull llama3
```

Start Radicale for appointment booking:

```bash
python -m radicale --storage-filesystem-folder=~/radicale/collections --auth-type none
```

Run the app:

```bash
streamlit run streamlit_app.py
```

Open the Streamlit URL, usually `http://localhost:8501`.
