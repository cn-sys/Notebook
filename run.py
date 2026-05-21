import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
from pathlib import Path

model_dir = Path('./model')
model_name = 'horizontal-text-detection-0001'
model_xml_path = model_dir / f'{model_name}.xml'

core = ov.Core()
model = core.read_model(model=str(model_xml_path))
compiled_model = core.compile_model(model, device_name='AUTO')

input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output('boxes')
N, C, H, W = input_layer_ir.shape

image = cv2.imread('intel_rnb.jpg')
resized_image = cv2.resize(image, (W, H))
input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

boxes = compiled_model([input_image])[output_layer_ir]
boxes = boxes[~np.all(boxes == 0, axis=1)]

def convert_result_to_image(bgr_image, resized_image, boxes, threshold=0.3):
    colors = {'red': (255, 0, 0), 'green': (0, 255, 0)}
    (real_y, real_x), (resized_y, resized_x) = (bgr_image.shape[:2], resized_image.shape[:2])
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    for box in boxes:
        conf = box[-1]
        if conf > threshold:
            x_min, y_min, x_max, y_max = [
                (int(max(corner_position * ratio_y, 10)) if idx % 2 else int(corner_position * ratio_x))
                for idx, corner_position in enumerate(box[:-1])
            ]
            rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors['green'], 3)
    return rgb_image

result = convert_result_to_image(image, resized_image, boxes)
plt.figure(figsize=(10, 6))
plt.axis('off')
plt.imshow(result)
plt.savefig('detection_result.png', bbox_inches='tight', pad_inches=0)
print(f'Done! Detected {len(boxes)} text regions.')
