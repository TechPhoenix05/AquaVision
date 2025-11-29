# ===================================
# ✅ SINGLE SPECIES IDENTIFICATION
# ===================================

# ==========================
# ✅ 1. IMPORTS
# ==========================
import tensorflow as tf
import json
import numpy as np
import cv2
from PIL import Image
import gradio as gr

# ==========================
# ✅ 2. LOAD MODEL & LABELS
# ==========================
model = tf.keras.models.load_model("/content/drive/MyDrive/MobileNet_Final/Ultimate_Final_Model.keras")

with open("/content/drive/MyDrive/class_labels.json", "r") as f:
    class_names = json.load(f)

print("✅ Ultimate Final Model & Classes Loaded!")

# ==========================
# ✅ 3. SEGMENTATION
# ==========================
def segment_microorganism(img):
    img = img.convert("RGB")
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform,
                               0.3 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_np, markers)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[markers > 1] = 255

    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img_np

    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)

    x = max(0, x - 10)
    y = max(0, y - 10)
    w = min(w + 20, img_np.shape[1] - x)
    h = min(h + 20, img_np.shape[0] - y)

    cropped = img_np[y:y+h, x:x+w]
    return cropped

# ==========================
# ✅ 4. RESIZE WITH PADDING
# ==========================
def resize_with_padding(img_np):
    h, w = img_np.shape[:2]
    scale = 224 / max(h, w)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(img_np, (new_w, new_h))

    delta_w = 224 - new_w
    delta_h = 224 - new_h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0,0,0)
    )
    return padded

# ==========================
# ✅ 5. PREPROCESS PIPELINE
# ==========================
IMG_SIZE = (224, 224)

def preprocess(img):
    seg = segment_microorganism(img)

    lab = cv2.cvtColor(seg, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    seg = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    seg = resize_with_padding(seg)

    seg = tf.keras.applications.mobilenet_v3.preprocess_input(seg)

    seg = np.expand_dims(seg, axis=0)
    return seg

# ==========================
# ✅ 6. PREDICTION (ENSEMBLE)
# ==========================
CONF_THRESHOLD = 0.70

def predict(img):
    img_array = preprocess(img)

    preds = []

    for angle in [0, 90, 180, 270]:
        rotated = np.rot90(img_array, k=angle//90, axes=(1,2))

        p1 = model.predict(rotated, verbose=0)[0]

        flipped = np.flip(rotated, axis=2)
        p2 = model.predict(flipped, verbose=0)[0]

        preds.append((p1 + p2) / 2)

    final_pred = np.mean(preds, axis=0)

    max_conf = np.max(final_pred)
    if max_conf < CONF_THRESHOLD:
        return {"Unsure – Needs Review": 1.0}

    sorted_idx = np.argsort(final_pred)[::-1]
    result = {}

    for i in range(3):
        cls = class_names[sorted_idx[i]]
        conf = float(final_pred[sorted_idx[i]])
        result[cls] = conf

    return result

# ==========================
# ✅ 7. GRADIO UI FOR SINGLE SPECIES IDENTIFICATION
# ==========================
def gradio_predict(img):
    try:
        if img is None:
            return "<b>⚠️ No image uploaded.</b>"

        result = predict(img)

        if isinstance(result, str):
            return f"<b>✅ {result}</b>"

        if isinstance(result, dict) and "Unsure – Needs Review" in result:
            return "<b>⚠️ Unsure – Needs Review (Confidence < 70%)</b>"

        html = "<h3>✅ Prediction Results</h3><table style='width:100%;text-align:left;'>"
        html += "<tr><th>Rank</th><th>Class</th><th>Confidence</th></tr>"

        rank = 1
        for cls, conf in result.items():
            html += f"<tr><td>{rank}</td><td>{cls}</td><td>{conf*100:.2f}%</td></tr>"
            rank += 1

        html += "</table>"
        return html

    except Exception as e:
        return f"<b>❌ Error:</b> {str(e)}"

demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.HTML(label="Result"),
    title="🔬 Ultimate Final Microorganism Classifier",
    description="""
✅ Auto-Segmentation
✅ Contrast Enhancement
✅ Top-3 Predictions
"""
)

demo.launch(debug=False)
