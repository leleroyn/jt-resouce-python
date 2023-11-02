from time import time
from typing import Dict, Any

from fastapi import APIRouter
from fastapi import UploadFile

from model.OcrResponse import OcrResponse
from service.ImageUtil import *
from service.Ocr import Ocr

app = APIRouter()


@app.post("/ocr")
async def ocr(image: UploadFile) -> Dict[str, Any]:
    start = time()
    image = image.file
    image = Image.open(image).convert('RGB')
    image = pil2cv(image)
    res = Ocr().detect(image)
    end = time()
    elapsed = end - start
    print('Elapsed time is %f seconds.' % elapsed)
    return OcrResponse(results=res).dict()
