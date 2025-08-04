from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,Session
from typing import Annotated
from fastapi import Depends
from redisconfig import settings

#POSTGRE DB CONNECTION
DATABASE_URL = 'postgresql://postgres:admin123@localhost/RAG'
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency=Annotated[Session, Depends(get_db)] 
