from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
def status():
    return {"status": "API is running"}

@router.get("/echo")
def echo(msg: str):
    return {"echo": msg}

@router.get("/square")
def square(x: int):
    return {"result": x * x}
