import cv2
import numpy as np
import onnxruntime as ort
import gradio as gr

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def output_to_keypoint(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        kpts = o[:, 6:]
        o = o[:, :6]
        for index, (*box, conf, cls) in enumerate(o):
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, *list(kpts[index])])
    return np.array(targets)


def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    # Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
        pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
        if steps == 3:
            conf1 = kpts[(sk[0] - 1) * steps + 2]
            conf2 = kpts[(sk[1] - 1) * steps + 2]
            if conf1 < 0.5 or conf2 < 0.5:
                continue
        if pos1[0] % 640 == 0 or pos1[1] % 640 == 0 or pos1[0] < 0 or pos1[1] < 0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0] < 0 or pos2[1] < 0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


# 定义预处理、推理和后处理的函数
def preprocess(img, model_shape=(960, 960)):
    meta = {"original": img}
    img, r, d = letterbox(img, model_shape, stride=64, auto=False)
    _img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
    _img = _img / 255.0
    _img = _img.astype(np.float32)
    meta.update({"resized": img, "norm": _img, "ratio": r, "pad": d})
    return meta

def postprocess(ort_outs, meta):
    output_kpt = output_to_keypoint(ort_outs)
    keypoints_x = output_kpt[:, 7:][:, ::3]
    keypoints_y = output_kpt[:, 7:][:, 1::3]
    keypoints_x = (keypoints_x - meta["pad"][0]) / meta["ratio"][0]
    keypoints_y = (keypoints_y - meta["pad"][1]) / meta["ratio"][1]
    output_kpt[:, 7:][:, ::3] = keypoints_x
    output_kpt[:, 7:][:, 1::3] = keypoints_y
    img_out = meta["original"].copy()
    for idx in range(output_kpt.shape[0]):
        plot_skeleton_kpts(img_out, output_kpt[idx, 7:].T, 3)
    return img_out

def predict(img, model):
    meta = preprocess(img)
    ort_inputs = {model.get_inputs()[0].name: meta["norm"]}
    ort_outs = model.run(["output"], ort_inputs)
    img_out = postprocess(ort_outs, meta)
    return img_out

# 初始化模型
model1 = ort.InferenceSession('model.onnx')  # 算法1模型
model2 = ort.InferenceSession('model.onnx')  # 算法2模型


def predict_algorithm1(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_out = predict(img, model1)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    return img_out

def predict_algorithm2(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_out = predict(img, model2)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    return img_out

# 创建 Gradio 界面
with gr.Blocks() as demo:
    # 主界面
    with gr.Row(visible=True) as main_interface:
        gr.Markdown("# 主界面")
        algo1_btn = gr.Button("进入算法1")
        algo2_btn = gr.Button("进入算法2")

    # 算法1界面
    with gr.Row(visible=False) as algo1_interface:
        gr.Markdown("## 算法1界面")
        image_input1 = gr.Image(type="pil", label="上传图片")
        image_output1 = gr.Image(type="pil", label="输出结果")
        back_btn1 = gr.Button("返回主界面")
        image_input1.change(predict_algorithm1, inputs=image_input1, outputs=image_output1)

    # 算法2界面
    with gr.Row(visible=False) as algo2_interface:
        gr.Markdown("## 算法2界面")
        image_input2 = gr.Image(type="pil", label="上传图片")
        image_output2 = gr.Image(type="pil", label="输出结果")
        back_btn2 = gr.Button("返回主界面")
        image_input2.change(predict_algorithm2, inputs=image_input2, outputs=image_output2)

    # 按钮逻辑
    algo1_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)),
        inputs=[],
        outputs=[main_interface, algo1_interface, algo2_interface],
    )
    algo2_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[main_interface, algo1_interface, algo2_interface],
    )
    back_btn1.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
        inputs=[],
        outputs=[main_interface, algo1_interface, algo2_interface],
    )
    back_btn2.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
        inputs=[],
        outputs=[main_interface, algo1_interface, algo2_interface],
    )

# 启动应用
if __name__ == "__main__":
    demo.launch()