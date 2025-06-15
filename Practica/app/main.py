from fastapi import FastAPI
from app import endpoints_basic, endpoints_hf

app = FastAPI(title="Practica Final FastAPI")

app.include_router(endpoints_basic.router)
app.include_router(endpoints_hf.router)
