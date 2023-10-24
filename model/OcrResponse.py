from copy import deepcopy
from typing import List, Dict, Any
from pydantic import BaseModel


class OcrResponse(BaseModel):
    status_code: int = 200
    results: List

    def dict(self, **kwargs):
        the_dict = deepcopy(super().dict())
        return the_dict
