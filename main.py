import gradio as gr
import onnxruntime as ort
import cv2
import numpy as np
from pose_estimation import predict_pose
from head_detection import predict_from_terminal
# 初始化模型
model1 = ort.InferenceSession('pose_estimation.onnx')  # 算法1模型
model2 = ort.InferenceSession('head_detection.onnx')  # 算法2模型

def predict_algorithm1(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_out = predict_pose(img, model1)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    return img_out

def predict_algorithm2(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_out = predict_from_terminal(img, model2)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    return img_out


with gr.Blocks() as demo:
    # 主界面 - 只在主界面显示标题
    with gr.Row(visible=True) as title_row:
        gr.Markdown('<div style="text-align: center; font-size: 30px; font-weight: bold;">主界面</div>')

    # 主界面 - 显示两个算法部分
    with gr.Row(visible=True) as main_interface:
        with gr.Column(elem_id="algo1_column"):
            algo1_image = gr.Image(value="pose_estimation.png", type="filepath", label="算法1")
            algo1_btn = gr.Button("姿态估计")
        with gr.Column(elem_id="algo2_column"):
            algo2_image = gr.Image(value="head_detection.png", type="filepath", label="算法2")
            algo2_btn = gr.Button("人头检测")

    # 算法1界面
    with gr.Row(visible=False) as algo1_interface:
        #gr.Markdown("## 姿态估计")
        image_input1 = gr.Image(type="pil", label="上传图片")
        image_output1 = gr.Image(type="pil", label="输出结果")
        back_btn1 = gr.Button("返回主界面")
        image_input1.change(predict_algorithm1, inputs=image_input1, outputs=image_output1)

    # 算法2界面
    with gr.Row(visible=False) as algo2_interface:
        #gr.Markdown("## 人头检测")
        image_input2 = gr.Image(type="pil", label="上传图片")
        image_output2 = gr.Image(type="pil", label="输出结果")
        back_btn2 = gr.Button("返回主界面")
        image_input2.change(predict_algorithm2, inputs=image_input2, outputs=image_output2)

    # 按钮逻辑
    algo1_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
        inputs=[],
        outputs=[main_interface, algo1_interface, algo2_interface, title_row],
    )
    algo2_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)),
        inputs=[],
        outputs=[main_interface, algo1_interface, algo2_interface, title_row],
    )
    back_btn1.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[main_interface, algo1_interface, algo2_interface, title_row],
    )
    back_btn2.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[main_interface, algo1_interface, algo2_interface, title_row],
    )
# 启动应用
if __name__ == "__main__":
    demo.launch()
