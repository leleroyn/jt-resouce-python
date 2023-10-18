from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Response
from time import time
from service.imageUtil import *
from service.Ocr import Ocr
from model.OcrResponse import OcrResponse

from fastapi import APIRouter
app = APIRouter()


@app.get("/")
async def root():
    return {"message": "Welcome to OCR Server!"}


@app.post("/ocr")
async def ocr(image: UploadFile) -> Dict[str, Any]:
    start = time()
    image = image.file
    image = Image.open(image).convert('RGB')
    image = rotate_image_by_exif(image)
    image = pil2cv(image)
    res = Ocr().detect(image)
    end = time()
    elapsed = end - start
    print('Elapsed time is %f seconds.' % elapsed)
    return OcrResponse(results=res).dict()


@app.post("/get_stamp")
async def get_stamp(image: UploadFile = File(...)):
    resp_str = ''
    image = image.file
    image = Image.open(image).convert('RGB')
    image = rotate_image_by_exif(image)
    cv_img = pil2cv(image)
    if check_seal_exit(cv_img) != 0:
        out_img = pick_seal_image(cv_img)
        resp_str = image_to_base64(cv2pil(out_img))
    return Response(content=resp_str, media_type="text/plain")