import uvicorn
from fastapi import FastAPI

from controller.ImageController import app as image_app
from controller.OcrController import app as ocr_app
from controller.PdfController import app as pdf_app

app = FastAPI()

app.include_router(ocr_app, tags=["ocr识别"])
app.include_router(image_app, tags=["图片处理"])
app.include_router(pdf_app, tags=["pdf处理"])


@app.get("/")
async def root():
    return {"message": "Welcome to OCR Server!"}


if __name__ == '__main__':
    uvicorn.run(app='main:app', host="0.0.0.0", port=8000, reload=True)
