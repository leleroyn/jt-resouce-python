from time import time
from typing import Dict, Any

from fastapi import APIRouter, Form
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
async def convent_pdf_to_image(file: UploadFile, merge: str = Form(default="0")):
    res = []
    images = convent_page_to_image(file.file.read())
    for i in range(len(images)):
        cur_img = Image.open(BytesIO(images[i]))
        cur_img = pil2cv(cur_img)
        if merge == "1":
            if i > 0:
                merge_img = cv2.hconcat(cur_img)
            else:
                merge_img = cur_img
            mask_text_on_bottom(str(i + 1) + "/" + str(len(images)), merge_img, (0, 0, 0))
            merge_img = cv2pil(merge_img)
            res.append(image_to_base64(merge_img))
        else:
            mask_text_on_bottom(str(i + 1) + "/" + str(len(images)), cur_img, (0, 0, 0))
            cur_img = cv2pil(cur_img)
            res.append(image_to_base64(cur_img))
    return res


@app.post("/adjust_image_position")
async def adjust_image_position(image: UploadFile, algorithm: str = Form(default="3")):
    """
    调整图像位置到正确的方向
    :param image:
    :param algorithm:
    :return:
    """
    image = image.file
    image = Image.open(image).convert('RGB')
    angle = 0
    if algorithm == "1":
        image = rotate_image_by_exif(image)
        return angle, image_to_base64(image)
    elif algorithm == "2":
        image = pil2cv(image)
        angle, image = orientation(image)
        image = cv2pil(image)
        return angle, image_to_base64(image)
    else:
        image = rotate_image_by_exif(image)
        image = pil2cv(image)
        angle, image = orientation(image)
        image = cv2pil(image)
        return angle, image_to_base64(image)


def get_stamp_single(image):
    image = rotate_image_by_exif(image)
    cv_img = pil2cv(image)
    out_img = pick_seal_image(cv_img)
    return out_img
