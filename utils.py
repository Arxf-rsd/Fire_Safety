import openvino as ov
import ipywidgets as widgets
import yaml
import cv2
import numpy as np
from ultralytics.utils.plotting import colors
from typing import Tuple
import time

# Initialize OpenVINO
core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value="AUTO",
    description="Device:",
    disabled=False,
)
device

# Load model
model = core.read_model(model="models/best.xml")
compiled_model = core.compile_model(model=model, device_name=device.value)

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
print("Input Layer shape:", input_layer.shape)
print("Output Layer shape:", output_layer.shape)

# Load labels
with open('models/metadata.yaml') as info:
    info_dict = yaml.load(info, Loader=yaml.Loader)
labels = info_dict['names']
print(labels)

# Letterbox function (resizes with padding)
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False,
              scale_fill=False, scaleup=False, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

# Preprocess function
def prepare_data(image, input_layer):
    input_w, input_h = input_layer.shape[2], input_layer.shape[3]
    input_image = letterbox(np.array(image), (input_w, input_h))[0]
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image / 255
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, 0)
    return input_image

# Inference test
image = cv2.imread("images/test.jpg")
if image is None:
    raise FileNotFoundError("Image not found at 'images/test.jpg'")

input_image = prepare_data(image, input_layer)
print("Input shape:", input_image.shape)
output = compiled_model([input_image])[output_layer]
print("Output shape:", output.shape)

# Postprocessing
def evaluate(output, conf_threshold):
    boxes, scores, label_key = [], [], []
    label_index = 0
    for class_ in output[0][4:]:
        for index in range(len(class_)):
            confidence = class_[index]
            if confidence > conf_threshold:
                xcen, ycen, w, h = output[0][0][index], output[0][1][index], output[0][2][index], output[0][3][index]
                xmin, xmax = int(xcen - (w/2)), int(xcen + (w/2))
                ymin, ymax = int(ycen - (h/2)), int(ycen + (h/2))
                boxes.append((xmin, ymin, xmax, ymax))
                scores.append(confidence)
                label_key.append(label_index)
        label_index += 1
    return np.array(boxes), np.array(scores), label_key

# NMS and IOU
def compute_iou(box, boxes, box_area, boxes_area):
    ys1 = np.maximum(box[0], boxes[:, 0])
    xs1 = np.maximum(box[1], boxes[:, 1])
    ys2 = np.minimum(box[2], boxes[:, 2])
    xs2 = np.minimum(box[3], boxes[:, 3])
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    unions = box_area + boxes_area - intersections
    return intersections / unions

def non_max_suppression(boxes, scores, conf_threshold):
    ys1, xs1 = boxes[:, 0], boxes[:, 1]
    ys2, xs2 = boxes[:, 2], boxes[:, 3]
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    keep = []
    while scores_indexes:
        index = scores_indexes.pop()
        keep.append(index)
        if not scores_indexes:
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index], areas[scores_indexes])
        scores_indexes = [v for i, v in enumerate(scores_indexes) if ious[i] <= conf_threshold]
    return np.array(keep)

# Draw results
def visualize(image, nms_output, boxes, label_key, scores):
    image_h, image_w = image.shape[:2]
    input_w, input_h = input_layer.shape[2], input_layer.shape[3]
    for i in nms_output:
        xmin, ymin, xmax, ymax = boxes[i]
        xmin = int(xmin * image_w / input_w)
        xmax = int(xmax * image_w / input_w)
        ymin = int(ymin * image_h / input_h)
        ymax = int(ymax * image_h / input_h)
        label = label_key[i]
        color = colors(label, True)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        text = f"{labels[label]} {scores[i]*100:.1f}%"
        cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return image

# Predict single image
def predict_image(image, conf_threshold):
    input_image = prepare_data(image, input_layer)
    start = time.time()
    output = compiled_model([input_image])[output_layer]
    end = time.time()
    boxes, scores, label_key = evaluate(output, conf_threshold)
    if len(boxes):
        nms_output = non_max_suppression(boxes, scores, conf_threshold)
        image = visualize(image, nms_output, boxes, label_key, scores)
    return image, end - start

# Test on single image
conf_threshold = 0.15
output_image, inference_time = predict_image(image, conf_threshold)
print(f"Inference time: {inference_time:.3f}s")
cv2.imshow("Detection Result", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Real-time webcam inference
def Main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Unable to open webcam.")
        return
    inference_times = []
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        visualized_frame, inference_time = predict_image(frame, conf_threshold)
        inference_times.append(inference_time)
        cv2.imshow("Press Spacebar to Exit", visualized_frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    avg_inf = np.mean(inference_times) * 1000
    print(f"Average inference time: {avg_inf:.2f} ms")
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()
