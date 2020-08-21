from pydantic import BaseModel


class QuestionProto(BaseModel):
    image_b64: str
    question: str


class ImageProto(BaseModel):
    image_b64: str
