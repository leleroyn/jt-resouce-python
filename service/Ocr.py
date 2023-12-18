import cv2
from paddleocr import PaddleOCR
from service.ImageUtil import orientation


class Ocr:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch")

    def detect(self, image):
        img_w = 1024 if image.shape[1] > 1024 else image.shape[1]
        image = cv2.resize(image, (img_w, int(img_w * image.shape[0] / image.shape[1])),
                           interpolation=cv2.INTER_AREA)
        angle, image = orientation(image)
        B_channel, G_channel, R_channel = cv2.split(image)
        result = self.ocr.ocr(R_channel, cls=True)
        if result is not None:
            results = []
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    item = {"position": line[0], "text": line[1][0], "score": line[1][1]}
                    print(line[1][0])
                    results.append(item)
        return results
