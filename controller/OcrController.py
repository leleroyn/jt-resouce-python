from time import time
from typing import Dict, Any

from fastapi import APIRouter
from fastapi import UploadFile, File, Response

from model.OcrResponse import OcrResponse
from service.ImageUtil import *
from service.Ocr import Ocr
from service.PdfUtils import *

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
async def get_stamp(image: List[UploadFile] = File(...)):
    img_list = []
    for img in image:
        img_list.append(get_stamp_single(Image.open(img.file).convert('RGB')))
    big_img = cv2.hconcat(img_list)
    resp_str = image_to_base64(cv2pil(big_img))
    return Response(content=resp_str, media_type="text/plain")


@app.post("/convent_pdf_to_image")
async def convent_pdf_to_image(file: UploadFile):
    res = []
    images = convent_page_to_image(file.file.read())
    for bits in images:
        res.append(bytes_to_base64(bits))
    return res


def get_stamp_single(image):
    image = rotate_image_by_exif(image)
    cv_img = pil2cv(image)
    out_img = pick_seal_image(cv_img)
    return out_img
