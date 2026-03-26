import hashlib
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, Depends, HTTPException
from fastapi.responses import FileResponse

from app.services.document_storage import get_storage_backend, DocumentStorageBackend
from app.services.document_registry import get_document_registry, BaseDocumentRegistry
from app.utils import now_ist_iso
from app.models import DocumentResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/", response_model=DocumentResponse, summary="Upload a new document")
async def upload_document(
    file: UploadFile = File(..., description="The physical PDF document file to upload."),
    title: Optional[str] = Form(None, description="Optional custom title for the document. Defaults to the filename if omitted."),
    description: Optional[str] = Form(None, description="Optional text describing the document's context or contents."),
    kb_name: Optional[str] = Form(None, description="Optional Knowledge Base name. Groups your files by providing a kb_name here for future vectorization."),
    storage: DocumentStorageBackend = Depends(get_storage_backend),
    registry: BaseDocumentRegistry = Depends(get_document_registry),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail={"error": "INVALID_FILE_TYPE", "message": "Only PDF files are supported."})

    file_bytes = await file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    existing = registry.find_by_hash(file_hash)
    if existing:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "DUPLICATE_FILE",
                "message": "This exact document has already been uploaded.",
                "existing_id": existing["id"],
            },
        )

    doc_id = str(uuid.uuid4())

    # Need to seek 0 because we read it into memory
    file.file.seek(0)
    storage.save(doc_id, file.filename, file.file)

    record = {
        "id": doc_id,
        "file_hash": file_hash,
        "original_filename": file.filename,
        "title": title or file.filename,
        "description": description,
        "kb_name": kb_name,
        "status": "UPLOADED",
        "file_size_bytes": len(file_bytes),
        "created_at": now_ist_iso(),
    }

    registry.add_record(record)
    return record


@router.get("/", response_model=list[DocumentResponse])
async def list_documents(registry: BaseDocumentRegistry = Depends(get_document_registry)):
    return registry.read_all()


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str, registry: BaseDocumentRegistry = Depends(get_document_registry)):
    record = registry.get_record(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail={"error": "NOT_FOUND", "message": "Document not found."})
    return record


@router.get("/{doc_id}/download")
async def download_document(doc_id: str, storage: DocumentStorageBackend = Depends(get_storage_backend), registry: BaseDocumentRegistry = Depends(get_document_registry)):
    record = registry.get_record(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail={"error": "NOT_FOUND", "message": "Document not found."})
    try:
        path = storage.get_path(doc_id)
        return FileResponse(path, filename=record["original_filename"])
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail={"error": "FILE_NOT_FOUND", "message": "Physical file not found."})


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, storage: DocumentStorageBackend = Depends(get_storage_backend), registry: BaseDocumentRegistry = Depends(get_document_registry)):
    record = registry.get_record(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail={"error": "NOT_FOUND", "message": "Document not found."})

    registry.delete_record(doc_id)
    storage.delete(doc_id)
    return {"status": "success", "message": "Document deleted."}
