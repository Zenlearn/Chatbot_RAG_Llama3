import uvicorn
from fastapi import FastAPI, UploadFile, Form, HTTPException, Query
from pre_processing.file_processing import extract_text, chunk_text
from services.translation import detect_language, translate
from services.embedding import generate_embedding
from database.vector_db_manager import add_document, query_db, delete_document
from llm.llm import prompt
from config.config_env import UVICORN_APP, PORT, RELOAD, HOST
from services.query_model import QueryRequest

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Document Search API!"}

@app.post("/upload")
async def upload_document(file: UploadFile,singleChunk: bool = Query(False),priority: int = Query(1)):
    """
    Upload a document (PDF or text) and store it in chunks after processing.
    singleChunk: If True, the document will be processed in a single chunk. [Default: False] {Query Parameter}
    priority: The priority of the document (1-5). [Default: 1] {Query Parameter}
    """
    try:
        if priority < 1 or priority > 5:
            raise Exception("Priority should be between 1 and 5")
        
        print(">",singleChunk, priority)
        # * Extract text from file (supports PDF and plain text)
        file_content = extract_text(file)
        
        print(len(file_content),">",file_content[:1000])

        # * Detect language of whole document
        detected_language = detect_language(file_content)

        # * Chunk the text into smaller parts for vectorization
        if singleChunk == True:
            # Todo: Implement to handle large files [can add text summarization]
            # if len(file_content) > 1000:
            #     raise Exception("File is too large to be processed in a single chunk")
            chunks = [file_content]
        else:
            chunks = chunk_text(file_content)

        # * Translate to English if necessary [Chunk by Chunk]
        if detected_language != "en":
            chunks = [translate(chunk) for chunk in chunks]

        # * Generate embeddings for each chunk
        embeddings = generate_embedding(chunks)

        # * Add chunks to the vector database [Qdrant]
        document = add_document(chunks, embeddings, priority)
        doc_id = document["doc_id"]
        chunks_ids = document["chunks_ids"]
        

        return {
            "message": "Document uploaded successfully",
            "doc_id": doc_id,
            "chunks": {
                "count": len(chunks),
                "ids": chunks_ids,
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
            query = translate(query)

        # * Generate embeddings for the query
        query_embedding = generate_embedding([query])[0]

        # * Query the database for similar vectors
        vectors = query_db(query_embedding)

        print(vectors)
        output = prompt(query, vectors)

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