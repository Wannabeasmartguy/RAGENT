from fastapi import Depends, FastAPI
from api.routers import chat
from api.routers import knowledgebase


app = FastAPI()


app.include_router(
    chat.router
)

app.include_router(
    knowledgebase.router
)