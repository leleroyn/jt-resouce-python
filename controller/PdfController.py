from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, Form

from service import convent_page_to_image, pil2cv, mask_text_on_bottom, cv2pil, image_to_base64

app = APIRouter()


@app.post("/convent_pdf_to_image")
async def convent_pdf_to_image(file: UploadFile, merge: str = Form(default="0")):
    """
    把pdf文件转换成图片
    :param file:  pdf文件
    :param merge: 是否合并成一张大图
    :return: 图片base64内容数组
    """
    res = []
    mask_images = []
    images = convent_page_to_image(file.file.read())
    merge = int(merge)
    top_img = pil2cv(Image.open(BytesIO(images[0])))
    dsize = (top_img.shape[1], top_img.shape[0])
    for i in range(len(images)):
        cur_img = Image.open(BytesIO(images[i]))
        cur_img = pil2cv(cur_img)
        mask_text_on_bottom(str(i + 1) + "/" + str(len(images)), cur_img, (0, 0, 0))
        cur_img = cv2.resize(cur_img, dsize=dsize, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        mask_images.append(cur_img)

    if merge == 1:
        separator = np.zeros((1, mask_images[0].shape[1], 3), dtype=np.uint8)
        separator[:, :, :] = [0, 0, 0]  # 设置分隔线颜色
        mask_images_with_seps = []
        for image in mask_images:
            height, width, _ = image.shape
            modified_image = np.concatenate((image, separator), axis=0)
            mask_images_with_seps.append(modified_image)
        merge_img = cv2.vconcat(mask_images_with_seps)
        merge_img = cv2pil(merge_img)
        res.append(image_to_base64(merge_img))
        return res
    else:
        for cur_img in mask_images:
            cur_img = cv2pil(cur_img)
            res.append(image_to_base64(cur_img))
        return res
