from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, Text,DateTime,ForeignKey
from db import Base,engine
from datetime import datetime

class ChatInput(BaseModel):
    user_message: str = Field(..., min_length=6, max_length=1000, 
                              description="User's query must be between 6 and 1000 characters.")

class Chats(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False) 
    intent = Column(Text, nullable=True)
    time_stamp = Column(DateTime(timezone=False), default=datetime.now)
    session_id = Column(Integer, ForeignKey("sessions.session_id", ondelete="CASCADE"))
  

class chat_sessions(Base):
    __tablename__ = "sessions"
    session_id = Column(Integer, primary_key=True, index=True)
    session_name = Column(Text, nullable=False)
    created_at=Column(DateTime(timezone=False), default=datetime.now) 

Base.metadata.create_all(bind=engine)    