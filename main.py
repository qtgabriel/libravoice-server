#Python modules
from fastapi import FastAPI, HTTPException

#Custom modules
from src.utils.data_model import *
from src.textToLibras.views import TextToLibras
from src.librasToText.views import LibrasToText

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/librasToText/translate")
async def ltt_translate(payload: LandmarkPayload):
    ltt = LibrasToText()
    try:
        result = ltt.process_points(payload.dict())
        return {"status": 200, "data": result}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/textToLibras/translate")
async def ttl_translate(data: TtlRequest):
    ttl = TextToLibras()
    try:
        translated_phrase = ttl.process_text(data.text)
        return {"status": 200, "data": translated_phrase}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
