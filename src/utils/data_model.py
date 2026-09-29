from pydantic import BaseModel

class TtlRequest(BaseModel):
    text: str