from fastapi import APIRouter, Request,UploadFile, File, Form,HTTPException
from fastapi.responses import HTMLResponse,JSONResponse,RedirectResponse, StreamingResponse
from urllib.parse import urlencode
from fastapi.templating import Jinja2Templates
from ragpipeline import get_formatted_bot_response,summarize_user_message 
from models import ChatInput,chat_sessions
from db_crud import create_chat, create_session, show_session_history
from db import db_dependency
from celery.result import AsyncResult
import os,io,csv,markdown,uuid
from typing import Dict, Any
from workers import process_csv_task, process_zip_task


templates = Jinja2Templates(directory="templates")


router = APIRouter()
@router.get("/", response_class=HTMLResponse)
async def home(request: Request, filename: str = None):
    filename = request.session.pop("filename", None)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "filename": filename
    })

@router.post("/chat",response_class=HTMLResponse)
async def process_chat(request:Request,db:db_dependency,prompt: str = Form(...)): 
    try:
        validated_input = ChatInput(user_message=prompt)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Message must be between 6 and 1000 characters.")
    
    botresponse = get_formatted_bot_response(validated_input.user_message,db=db)
    session_name= summarize_user_message(validated_input.user_message)
    intent = botresponse["intent"]
    response=markdown.markdown(botresponse["response"])    
    sessionid = create_session(session_name,db)
    create_chat(validated_input.user_message, response, intent,sessionid,db)
    return templates.TemplateResponse("chatbot.html", {
        "request": request,
        "initial_bot_response": response,
        "session_id": sessionid,
    })  


@router.get("/history", response_class=HTMLResponse)
async def show_history(request: Request,db:db_dependency):
    dbquery = show_session_history(db)
    return templates.TemplateResponse("session-history.html", {
        "request": request,
        "sessions": dbquery
    })
@router.post("/chat-history", response_class=HTMLResponse)
async def show_history(request: Request,db:db_dependency,session_id:str=Form(...)):
    request.session['session_id']=session_id
    dbquery = show_session_history(db,session_id=request.session.get('session_id'))
    for chat in dbquery:
        chat.user_message = markdown.markdown(chat.user_message)
        chat.bot_response = markdown.markdown(chat.bot_response)
    return templates.TemplateResponse("chat-history.html", {
        "request": request,
        "session": dbquery,
    })
@router.post("/botchat")
async def process_chat(db:db_dependency,session_id:int=Form(...),botprompt:str= Form(...)):
    
    try:
        validated_input = ChatInput(user_message=botprompt)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Message must be between 6 and 1000 characters.")
    
    session = db.query(chat_sessions).filter(chat_sessions.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found") 
    
    botresponse = get_formatted_bot_response(validated_input.user_message,db=db,session_id=session_id)
    intent = botresponse["intent"]
    response=botresponse["response"] 
    create_chat(validated_input.user_message,response,intent,session_id,db)
    return JSONResponse({
        "response": response,
    })
@router.post("/upload",response_class=HTMLResponse)
async def upload_document(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    UPLOAD_DIR = "uploads"
    EXTRACTED_DIR = "uploads/extracted_csvs"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    # Validate file extension
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in ('.csv', '.zip'):
        raise HTTPException(400, "Only CSV and ZIP files are supported")

    # Create a unique file name to avoid collisions
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    # Save file directly to /uploads
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Dispatch to Celery
    if ext == '.csv':
        task = process_csv_task.delay(save_path, filename)
    else:
        task = process_zip_task.delay(save_path, filename)

    # Optionally track this upload session
    request.session["filename"] = filename

    # Redirect to home (or another page)
    return RedirectResponse(url="/", status_code=303)

@router.get("/status/{task_id}")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """Check status of a processing task"""
    task = AsyncResult(task_id)
    
    if task.failed():
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(task.result)
        }
    
    return {
        "task_id": task_id,
        "status": task.status,
        "progress": task.info if task.status == "PROGRESS" else None,
        "result": task.result if task.ready() else None
    }

@router.post("/download_csv")
async def create_csv(request:Request,db:db_dependency,session_id:str = Form(...)):
    session_id = request.session.get("session_id")
    dbquery = show_session_history(db,session_id=session_id)
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["User Message","Bot Response", "Intent"])
    for chat in dbquery:
        writer.writerow([chat.user_message,chat.bot_response,chat.intent])
    output.seek(0)  
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=chat_logs.csv"}
    )
