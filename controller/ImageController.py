from typing import List

import cv2
from PIL import Image
from fastapi import APIRouter, UploadFile, Form, File
from fastapi.openapi.models import Response

from service import rotate_image_by_exif, pil2cv, pick_seal_image, image_to_base64, orientation, cv2pil

app = APIRouter()


@app.post("/get_stamp")
async def get_stamp(image: List[UploadFile] = File(...)):
    """
    提取上传图片的印章
    :param image: 上传的图片，支持多张图片
    :return:图片base64内容
    """
    img_list = []
    for img in image:
        img_list.append(get_stamp_single(Image.open(img.file).convert('RGB')))
    big_img = cv2.hconcat(img_list)
    resp_str = image_to_base64(cv2pil(big_img))
    return Response(content=resp_str, media_type="text/plain")


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
