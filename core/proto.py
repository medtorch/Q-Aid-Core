from pydantic import BaseModel


class QuestionProto(BaseModel):
    image_b64: str
    question: str
    category: str


class ImageProto(BaseModel):
    image_b64: str
