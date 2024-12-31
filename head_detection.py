import onnxruntime
import cv2
import numpy as np

# Global Variables
confidence = 80
conf_thresold = 0.8
iou_thresold = 0.3
Display_Confidence = True
Display_Class = True

# Load image
def load_image(image_path, input_shape):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_height, input_width = input_shape[2:]
    resized = cv2.resize(rgb_image, (input_width, input_height))
    # Scale input pixel value to 0 to 1
    input_image = resized / 255.0
    input_image = input_image.transpose(2, 0, 1)
    input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)

    return [image, input_tensor, rgb_image]

# Load model
def load_model(model_path):
    opt_session = onnxruntime.SessionOptions()
    opt_session.enable_mem_pattern = False
    opt_session.enable_cpu_mem_arena = False
    opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    EP_list = ['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)
    model_inputs = ort_session.get_inputs()
    input_shape = model_inputs[0].shape

    return [ort_session, input_shape]

# Run inference using the ONNX model
def predict(image, ort_session, input_tensor):
    global conf_thresold

    model_inputs = ort_session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    input_shape = model_inputs[0].shape
    input_height, input_width = input_shape[2:]
    image_height, image_width = image.shape[:2]
    model_output = ort_session.get_outputs()
    output_names = [model_output[i].name for i in range(len(model_output))]
    outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]
    predictions = np.squeeze(outputs).T

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_thresold, :]
    scores = scores[scores > conf_thresold]
    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    # Get bounding boxes for each object
    boxes = predictions[:, :4]
    # Rescale boxes
    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_width, image_height, image_width, image_height])
    boxes = boxes.astype(np.int32)

    return [boxes, scores, class_ids]

# Annotate image
def annotate(image, boxes, scores, class_ids):
    global iou_thresold
    global Display_Confidence
    global Display_Class

    indices = nms(boxes, scores, iou_thresold)

    # Define classes
    CLASSES = ['head']
    image_draw = image.copy()
    for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
        bbox = bbox.round().astype(np.int32).tolist()
        cls_id = int(label)
        cls = CLASSES[cls_id]
        x1, y1, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        display_message = ""
        if Display_Class:
            display_message = display_message + cls
        if Display_Confidence:
            display_message = f"{display_message} {score:.2f}"
        cv2.rectangle(image_draw, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 1)
        if Display_Confidence or Display_Class:
            (text_width, text_height), _ = cv2.getTextSize(display_message, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(image_draw, (x1, y1), (x1 + text_width, y1 - text_height - 10), (0, 255, 0), -1)
            cv2.putText(image_draw, display_message, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    rgb_image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
    return rgb_image_draw

# Non-Maximum Suppression
def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

# Compute Intersection over Union
def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    iou = intersection_area / union_area

    return iou

# Convert bounding boxes
def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

# Main prediction function
def prediction(image_path, conf=80, disp_Class=True, disp_Confidence=True,
               iou_thresh_=30, model_path="model.onnx"):
    global confidence
    global conf_thresold
    global iou_thresold
    global Display_Confidence
    global Display_Class

    Display_Confidence = disp_Confidence
    Display_Class = disp_Class
    confidence = conf
    conf_thresold = confidence / 100
    iou_thresold = iou_thresh_

    model = load_model(model_path)
    input_I = load_image(image_path, model[1])
    predictions = predict(input_I[0], model[0], input_I[1])
    annotated_image = annotate(input_I[0], predictions[0], predictions[1], predictions[2])

    return annotated_image

# Predict from terminal
def predict_from_terminal(image, ort_session):
    input_shape = ort_session.get_inputs()[0].shape
    input_height, input_width = input_shape[2:]

    # Preprocess image
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_image, (input_width, input_height))
    input_image = resized / 255.0
    input_image = input_image.transpose(2, 0, 1)
    input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)

    # Predict
    predictions = predict(image, ort_session, input_tensor)
    annotated_image = annotate(image, predictions[0], predictions[1], predictions[2])

    return annotated_image
