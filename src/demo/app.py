import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import time
from pathlib import Path
import io

st.set_page_config(page_title="QuantLab-8bit Demo", layout="centered")

st.title("QuantLab-8bit — FP32 vs INT8 (ORT CPU)")
st.write("Upload an image and compare FP32 vs INT8 predictions and latency side-by-side.")

# Sidebar: model paths
st.sidebar.header("Models")
fp32_path = st.sidebar.text_input("FP32 ONNX path", "artifacts/onnx/fp32/mobilenet_v2_cifar10/model.onnx")
int8_path = st.sidebar.text_input("INT8 ONNX path", "artifacts/onnx/ptq/mobilenet_v2_cifar10/pcw_symW_asymA_minmax/model.onnx")

# Sidebar: input size
st.sidebar.header("Input size (C,H,W)")
c = st.sidebar.number_input("C", value=3, step=1)
h = st.sidebar.number_input("H", value=32, step=1)
w = st.sidebar.number_input("W", value=32, step=1)

uploaded = st.file_uploader("Upload an image (RGB)", type=["png","jpg","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((int(w), int(h)))
    st.image(img, caption="Preprocessed input", use_container_width=True)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2,0,1))  # CHW
    arr = arr[None, ...]              # NCHW batch=1

    def run_sess(path):
        if not Path(path).exists():
            return None, None, f"Missing: {path}"
        so = ort.SessionOptions()
        sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        # warmup
        for _ in range(5):
            _ = sess.run([out_name], {in_name: arr})
        # measure
        t0 = time.perf_counter()
        y = sess.run([out_name], {in_name: arr})[0]
        ms = (time.perf_counter() - t0) * 1000.0
        probs = np.exp(y - y.max()) / np.exp(y - y.max()).sum()
        top1 = int(probs.argmax())
        conf = float(probs.max())
        return top1, (conf, ms), None

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("FP32")
        k, info, err = run_sess(fp32_path)
        if err: st.error(err)
        else:
            st.write(f"Top-1: **{k}**  ·  conf={info[0]:.3f}  ·  latency={info[1]:.2f} ms")
    with col2:
        st.subheader("INT8")
        k, info, err = run_sess(int8_path)
        if err: st.error(err)
        else:
            st.write(f"Top-1: **{k}**  ·  conf={info[0]:.3f}  ·  latency={info[1]:.2f} ms")

    st.caption("Note: class indices are numeric for simplicity. Map them to labels if you wish.")
else:
    st.info("Upload an image to run the comparison.")