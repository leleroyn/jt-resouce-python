from fastapi import FastAPI
import uvicorn
from controller.OcrController import app as ocr_app

app = FastAPI()

app.include_router(ocr_app,tags=["文本识别","印章提取"])

if __name__ == '__main__':
    uvicorn.run(app='main:app', host="0.0.0.0", port=8000, reload=True)
