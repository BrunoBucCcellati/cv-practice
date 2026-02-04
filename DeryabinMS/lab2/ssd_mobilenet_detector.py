from typing import List, Tuple
import numpy as np

from detector_base import BaseDetector, Detection, DetectorConfig


class SSDMobileNetDetector(BaseDetector):
    """Детектор на основе SSD MobileNet"""

    def _postprocess(self, outputs,
                     image_shape: Tuple[int, int]) -> List[Detection]:
        img_h, img_w = image_shape
        detections: List[Detection] = []

        for det in outputs[0, 0]:
            confidence = float(det[2])
            if confidence < self.config.conf_threshold:
                continue
            class_id = int(det[1])

            x1 = int(det[3] * img_w)
            y1 = int(det[4] * img_h)
            x2 = int(det[5] * img_w)
            y2 = int(det[6] * img_h)

            if 1 <= class_id < len(self.config.classes) + 1:
                class_name = self.config.classes[class_id - 1]
            else:
                class_name = f"class_{class_id - 1}"

            detections.append(
                Detection(class_id=class_id,
                          class_name=class_name,
                          confidence=confidence,
                          bbox=(x1, y1, x2, y2)))

        return detections
