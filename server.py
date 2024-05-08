from fastapi import Depends, FastAPI
from api.routers import chat


app = FastAPI()


app.include_router(
    chat.router
)