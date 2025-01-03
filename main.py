import uvicorn
from fastapi import FastAPI, UploadFile, Form, HTTPException, Query
from pre_processing.file_processing import extract_text, chunk_text
from services.translation import detect_language, translate
# from services.embedding import generate_embedding #? Not required in ChromaDB
# from database.qdrant_vector_db import add_document, query_db, delete_document #? Not required in ChromaDB
from database.chromadb_vector_db import add_document, query_db, delete_document
from llm.llm import prompt
from config.config_env import UVICORN_APP, PORT, RELOAD, HOST
from services.query_model import QueryRequest
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://suryansh-dey.github.io/",
    "https://zenlearn.ai"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "success": True,
        "message": "Zenlearn AI coach: Your personalized guide to actionable insights and lifelong learning",
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile, singleChunk: bool = Query(False), priority: int = Query(1)
):
    """
    Upload a document (PDF or text) and store it in chunks after processing.
    singleChunk: If True, the document will be processed in a single chunk. [Default: False] {Query Parameter}
    priority: The priority of the document (1-5). [Default: 1] {Query Parameter}
    """
    try:
        if priority < 1 or priority > 5:
            raise Exception("Priority should be between 1 and 5")

        # * Extract text from file (supports PDF and plain text)
        file_content = extract_text(file)

        print(len(file_content),">",file_content[:1000])

        # * Detect language of whole document
        detected_language = detect_language(file_content)
        
        print("Detected Language:", detected_language)

        # * Chunk the text into smaller parts for vectorization
        if singleChunk == True:
            # Todo: Implement to handle large files [can add text summarization]
            if len(file_content) >= 256:
                raise Exception(
                    "File is too large to be processed in a single chunk, please set singleChunk to False"
                )
            chunks = [file_content]
        else:
            chunks = chunk_text(file_content)
        
        # * Translate to English if necessary [Chunk by Chunk]
        if detected_language != "en":
            #! Translation is disabled for now [Due to unavailablity of BHASHINI API key]
            # chunks = [translate(chunk) for chunk in chunks]
            pass

        # * Generate embeddings for each chunk
        # embeddings = generate_embedding(chunks)

        print("no of Chunks:", len(chunks))
        # * Add chunks to the vector database [Qdrant]
        document = add_document(chunks, priority)
        print("Document:", document)
        doc_id = document["doc_id"]
        # chunks_ids = document["chunks_ids"] #? Not required in ChromaDB

        return {
            "message": "Document uploaded successfully",
            "doc_id": doc_id,
            "chunks": {
                "count": len(chunks),
                # "ids": chunks_ids, #? Not required in ChromaDB
            },
            "detected_language": detected_language,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_document(body: QueryRequest):
    """
    Query the database and return relevant results.
    """
    try:
        query = body.query

        # * Detect language of query
        detected_language = detect_language(query)

        # * Translate to English if necessary
        if detected_language != "en":
            #! Translation is disabled for now [Due to unavailablity of BHASHINI API key]
            # query = translate(query)
            pass

        # * Generate embeddings for the query
        # query_embedding = generate_embedding([query])[0]

        # * Query the database for similar vectors
        vectors = query_db(query)

        # print("Vectors:", vectors)
        output = prompt(query, vectors)
        # print("Output:", output)

        return {
            "message": "Query successful",
            "output": output,
            "detected_language": detected_language,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete/{doc_id}")
async def delete_document_route(doc_id: str):
    """
    Delete a document by its ID.
    """
    try:
        if not doc_id:
            raise Exception("Document ID is required")
        metadata = delete_document(doc_id)
        return {
            "message": "Document deleted successfully",
            "success": True,
            "data": metadata,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(UVICORN_APP, host=HOST, port=PORT, reload=RELOAD)
