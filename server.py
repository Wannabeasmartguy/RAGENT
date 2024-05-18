from fastapi import Depends, FastAPI
from api.routers import chat
from api.routers import knowledgebase
from api.routers import agentchat


app = FastAPI()


app.include_router(
    chat.router
)

app.include_router(
    knowledgebase.router
)

app.include_router(
    agentchat.router 
)