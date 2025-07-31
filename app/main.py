from fastapi import FastAPI
from routes import router
from starlette.middleware.sessions import SessionMiddleware



app = FastAPI()
app.include_router(router)
app.add_middleware(SessionMiddleware, secret_key="your_secret_key_here")

      