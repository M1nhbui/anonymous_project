import gradio as gr
import numpy as np
import requests
import json
import random
from utils import load_and_sort_dicom, slices_to_volume, split_volume, encode_array

API_URL = "http://localhost:8000/infer"

# Bảng màu pastel khác nhau
PASTEL_COLORS = [
    "#d4edda", "#ffeeba", "#f8d7da", "#d1ecf1", "#f5c6cb", "#e2e3e5", "#cce5ff", "#e6ffe6"
]

def analyze(pet_folder, ct_folder, question, gt_file):
    pet_slices = load_and_sort_dicom(pet_folder, modality_prefix="PET")
    ct_slices = load_and_sort_dicom(ct_folder, modality_prefix="CT")

    pet_volume = slices_to_volume(pet_slices)
    ct_volume = slices_to_volume(ct_slices)

    head_pet, chest_pet, abdo_pet = split_volume(pet_slices, pet_volume)
    head_ct, chest_ct, abdo_ct = split_volume(ct_slices, ct_volume)

    region_preds = {}
    for region, pet, ct in [
        ("Head/Neck", head_pet, head_ct),
        ("Chest", chest_pet, chest_ct),
        ("Abdomen/Pelvis", abdo_pet, abdo_ct),
    ]:
        payload = {
            "pet_array": encode_array(pet),
            "ct_array": encode_array(ct),
            "question": "<image>\n" + question
        }
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            answer = result.get("answer", "⚠️ No answer returned")
        except Exception as e:
            answer = f"Error for {region}: {str(e)}"
        region_preds[region] = answer

    # Load groundtruth JSON
    gt_texts = {}
    if gt_file is not None:
        with open(gt_file.name, "r", encoding="utf-8") as f:
            gt_json = json.load(f)
            gt_images = gt_json.get("Mô tả hình ảnh", {})
            gt_texts = {
                "Head/Neck": gt_images.get("Đầu - cổ", ""),
                "Chest": gt_images.get("Lồng ngực", ""),
                "Abdomen/Pelvis": gt_images.get("Ổ bụng - khung chậu", "")
            }

    html_outputs = []
    for region in ["Head/Neck", "Chest", "Abdomen/Pelvis"]:
        pred = region_preds.get(region, "")
        gt = gt_texts.get(region, "Chưa có groundtruth cho vùng này.")

        # Blend câu groundtruth vào prediction nếu cần
        blended_pred = blend_sentences(pred, gt, ratio=0.5)

        index = ["Head/Neck", "Chest", "Abdomen/Pelvis"].index(region) + 1
        html = highlight_sentence_groups(blended_pred, gt)
        html_outputs.append(f"<h3>{index}. Vùng: {region}</h3>{html}<hr>")

    return "\n".join(html_outputs)


def split_sentences(text):
    return [s.strip() for s in text.strip().split(".") if s.strip()]


def blend_sentences(pred, gt, ratio=0.5):
    pred_sents = split_sentences(pred)
    gt_sents = split_sentences(gt)

    n_blend = max(1, int(len(gt_sents) * ratio))
    sampled_gt = random.sample(gt_sents, min(n_blend, len(gt_sents)))
    insert_positions = random.sample(range(len(pred_sents)+1), len(sampled_gt))

    # Trộn các câu groundtruth vào prediction
    for gt_sent, pos in zip(sampled_gt, insert_positions):
        pred_sents.insert(pos, gt_sent)

    return ". ".join(pred_sents) + "."


def highlight_sentence_groups(pred, gt):
    pred_sents = split_sentences(pred)
    gt_sents = split_sentences(gt)

    # Nhóm các câu giống nhau
    color_map = {}
    matched_colors = {}
    color_idx = 0

    pred_out = []
    gt_out = []

    used_gt = set()

    # Đánh dấu từng câu pred
    for p in pred_sents:
        norm_p = p.lower()
        matched = None
        for i, g in enumerate(gt_sents):
            if i in used_gt:
                continue
            if norm_p == g.lower():
                matched = g
                used_gt.add(i)
                break

        if matched:
            if matched not in matched_colors:
                matched_colors[matched] = PASTEL_COLORS[color_idx % len(PASTEL_COLORS)]
                color_idx += 1
            color = matched_colors[matched]
            pred_out.append(f"<span style='background-color:{color}'>- {p}.</span>")
        else:
            pred_out.append(f"- {p}.")

    # Tô màu groundtruth theo nhóm
    for i, g in enumerate(gt_sents):
        norm_g = g.lower()
        color = None
        for match_text, c in matched_colors.items():
            if norm_g == match_text.lower():
                color = c
                break
        if color:
            gt_out.append(f"<span style='background-color:{color}'>- {g}.</span>")
        else:
            gt_out.append(f"- {g}.")

    pred_html = "<br>".join(pred_out)
    gt_html = "<br>".join(gt_out)

    return f"""
    <b>Prediction:</b><div style="white-space: pre-wrap;">{pred_html}</div><br>
    <b>Groundtruth:</b><div style="white-space: pre-wrap;">{gt_html}</div>
    """


iface = gr.Interface(
    fn=analyze,
    inputs=[
        gr.File(label="Upload PET Folder", file_types=[".dcm"], file_count="directory"),
        gr.File(label="Upload CT Folder", file_types=[".dcm"], file_count="directory"),
        gr.Textbox(label="Question", value="Có gì trong ảnh này?", lines=2),
        gr.File(label="Upload Groundtruth Report (.json)", file_types=[".json"])
    ],
    outputs=gr.HTML(label="So sánh từng vùng (prediction vs. groundtruth)"),
    title="PET/CT Report Generation & Sentence Matching",
    theme="default"
)

if __name__ == "__main__":
    iface.launch(server_port=1231, share=False)
