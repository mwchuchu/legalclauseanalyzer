from models import Chats,chat_sessions
from db import db_dependency
from fastapi import HTTPException, Depends



def create_chat(user_message:str,bot_response:str,intent:str,session_id:int, db:db_dependency):
    try:
        chat_entry = Chats(
            user_message=user_message,
            bot_response=bot_response,
            intent=intent,
            session_id=session_id,
        )
        db.add(chat_entry)
        db.commit()
        db.refresh(chat_entry)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
def create_session(session_name:str, db:db_dependency):
    try:
        session_entry = chat_sessions(
            session_name=session_name,
        )
        db.add(session_entry)
        db.commit()
        db.refresh(session_entry)
        return session_entry.session_id
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
def show_session_history(db: db_dependency, session_id: int = None):
    try:
        if session_id:
            history = db.query(Chats).filter(Chats.session_id == session_id).order_by(Chats.time_stamp.desc()).all()
        else:
            history = db.query(chat_sessions).order_by(chat_sessions.created_at.desc()).all()   
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Helper function with improved chat history
def get_follow_up_chats(db: db_dependency, session_id: int) -> str:
    """Enhanced version with error handling and structured output"""
    try:
        chats = db.query(Chats)\
                .filter(Chats.session_id == session_id)\
                .order_by(Chats.time_stamp.desc())\
                .limit(3)\
                .all()
                
        if not chats:
            return None
            
        return "\n".join(
            f"Q{i+1}: {chat.user_message}\n"
            f"A{i+1}: {chat.bot_response}\n"
            for i, chat in enumerate(chats)
        )
        
    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        return None


