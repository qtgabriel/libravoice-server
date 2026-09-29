#Python modules
from fastapi import FastAPI

#Custom modules
from src.utils.data_model import *
from src.textToLibras.views import TextToLibras

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/librasToText/translate")
async def ltt_translate(text: str):
    return

@app.post("/textToLibras/translate")
async def ttl_translate(data: TtlRequest):
    ttl = TextToLibras()
    try:
        translated_phrase = ttl.process_text(data.text)
        return {"status": 200, "data": translated_phrase}
    except Exception as e:
        print(e)
        return 500
