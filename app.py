# -*- coding: utf-8 -*-
# app.py — Dog Behavior & Affect Analyzer (Cloud-safe, no-audio)
# 说明：移除了 librosa/scipy 依赖，音频/吠叫检测降级为禁用状态（AUDIO_OK=False）
# 功能：视频上传 -> YOLOv8 检测 -> 时序特征 -> 行为识别(增量学习) -> 情绪映射 -> 时间轴/报告/主动学习

import os, io, json, time, uuid, math, tempfile
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import streamlit as st
import cv2
from ultralytics import YOLO
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
import joblib

# ---------------- 全局配置 ----------------
APP_TITLE = "🐶 Dog Behavior & Affect Analyzer"
DATA_DIR = "data_samples"
MODEL_DIR = "models"
REPORT_DIR = "reports"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

LABELS = ["lying", "sitting/idle", "walking", "running", "sprinting/jumping"]
AFFECT_TABLE = {
    "lying": (0.20, 0.70),           # (arousal, valence)
    "sitting/idle": (0.30, 0.60),
    "walking": (0.45, 0.65),
    "running": (0.70, 0.65),
    "sprinting/jumping": (0.85, 0.60),
}

# 音频分支关闭（无 librosa/scipy）
AUDIO_OK = False

# ---------------- 数据结构 ----------------
@dataclass
class Segment:
    seg_id: str
    t_start: float
    t_end: float
    features: np.ndarray
    auto_label: str
    auto_conf: float
    bark: bool  # 这里占位，当前恒为 False（无音频）

# ---------------- 模型加载 ----------------
@st.cache_resource
def load_detector():
    # 轻量 YOLOv8，首次会自动下载权重
    return YOLO("yolov8n.pt")

# ---------------- 工具函数 ----------------
def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0])*(a[3]-a[1]); areaB = (b[2]-b[0])*(b[3]-b[1])
    union = areaA + areaB - inter + 1e-6
    return inter / union

def extract_audio_signal(video_path, target_sr=16000):
    # 无音频依赖时，直接返回空
    return None, None

def bark_score_track(y, sr, frame_ms=400, hop_ms=160):
    # 无音频依赖时，直接返回空
    return []

def rule_behavior(speed_px, aspect_ratio, area_change):
    # 简单规则：速度主导，形态修正
    if speed_px < 2.0:
        if aspect_ratio < 0.85 and area_change < 0.01:
            return "lying", 0.70
        return "sitting/idle", 0.60
    elif speed_px < 10.0:
        return "walking", 0.70
    elif speed_px < 23.0:
        return "running", 0.75
    else:
        return "sprinting/jumping", 0.80

def affect_from_behavior(label:str, bark:bool):
    a, v = AFFECT_TABLE.get(label, (0.5, 0.5))
    # 无音频分支，不对唤醒度加成
    conf_aff = 0.45 if label in ["lying","sitting/idle"] else 0.55
    return a, v, conf_aff

# --------- 增量学习：存取样本与训练 ---------
def save_sample(features: np.ndarray, true_label: str, meta: dict):
    sid = str(uuid.uuid4())
    np.save(os.path.join(DATA_DIR, f"{sid}_x.npy"), features.astype(np.float32))
    json.dump({"y": true_label, "meta": meta}, open(os.path.join(DATA_DIR, f"{sid}_y.json"), "w"))

def load_samples(limit=None):
    xs, ys = [], []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith("_y.json")]
    if limit: files = files[:limit]
    for jf in files:
        meta = json.load(open(os.path.join(DATA_DIR, jf)))
        y = meta["y"]
        sid = jf.replace("_y.json","")
        x = np.load(os.path.join(DATA_DIR, f"{sid}_x.npy"))
        xs.append(x); ys.append(y)
    if not xs: return None, None
    X = np.vstack(xs)
    y = np.array([LABELS.index(v) for v in ys], dtype=np.int64)
    return X, y

def fit_or_partial_update(X_train, y_train):
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_train)
    Xs = scaler.transform(X_train)

    base = SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="optimal", random_state=42)
    classes = np.arange(len(LABELS))
    base.partial_fit(Xs, y_train, classes=classes)

    clf = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
    clf.fit(Xs, y_train)

    ts = time.strftime("%Y%m%d_%H%M%S")
    joblib.dump(clf, os.path.join(MODEL_DIR, f"behavior_clf_{ts}.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{ts}.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "behavior_clf_latest.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_latest.joblib"))
    return clf, scaler, ts

def predict_with_model(features_vec: np.ndarray):
    model_p = os.path.join(MODEL_DIR, "behavior_clf_latest.joblib")
    scaler_p = os.path.join(MODEL_DIR, "scaler_latest.joblib")
    if not (os.path.exists(model_p) and os.path.exists(scaler_p)):
        return None
    clf = joblib.load(model_p)
    scaler = joblib.load(scaler_p)
    Xs = scaler.transform(features_vec.reshape(1, -1))
    probs = clf.predict_proba(Xs)[0]
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)
st.caption("上传短视频，获得：行为识别（含置信度）与情绪（唤醒/效价）推断。当前为 Cloud 安全版，音频吠叫检测已禁用。")

with st.sidebar:
    st.header("参数")
    max_seconds = st.slider("分析时长上限(秒)", 5, 90, 25)
    conf_th = st.slider("检测置信度阈值（YOLO）", 0.1, 0.8, 0.35)
    sample_fps = st.slider("分析抽帧速率(fps)", 3, 12, 6)
    lowconf_th = st.slider("低置信度阈值（触发标注）", 0.50, 0.90, 0.65)
    st.markdown("---")
    if st.button("🧠 使用已标注样本改进模型"):
        X_all, y_all = load_samples()
        if X_all is None:
            st.warning("暂无标注样本。先在下方时间轴中保存几条训练样本。")
        else:
            _, _, tag = fit_or_partial_update(X_all, y_all)
            st.success(f"模型已更新 ✅（版本 {tag}）")

uploaded = st.file_uploader("上传视频 (mp4/mov/mkv)", type=["mp4","mov","mkv"])

if uploaded:
    # 临时保存
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmpf.write(uploaded.read()); tmpf.close()

    # 无音频：占位空轨迹
    bark_track = []
    def bark_present(t0, t1):
        return (False, 0.0)

    det = load_detector()
    cap = cv2.VideoCapture(tmpf.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), max_seconds*fps))
    step = max(1, int(round(fps / sample_fps)))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.info("开始分析（为提速采用抽帧）。")
    progress = st.progress(0.0)

    last_box = None
    last_area = None
    segments: List[Segment] = []
    window_frames = max(3, int(sample_fps * 1.2))  # ~1.2s
    buf_feats, buf_times = [], []

    for idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        t_sec = idx / fps

        res = det(frame, conf=conf_th, verbose=False)[0]
        dog_boxes = []
        for b in res.boxes:
            cls = int(b.cls[0].item())
            if det.model.names.get(cls, "") == "dog":
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                dog_boxes.append([x1,y1,x2,y2, float(b.conf[0].item())])

        if dog_boxes:
            dog_boxes.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
            box = dog_boxes[0][:4]

            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            w_box, h_box = box[2]-box[0], box[3]-box[1]
            area = w_box * h_box
            aspect = min(w_box, h_box) / (max(w_box, h_box)+1e-6)

            speed = 0.0; acc = 0.0; area_chg = 0.0
            if last_box is not None and iou(last_box, box) > 0.1:
                lx, ly = (last_box[0]+last_box[2])/2, (last_box[1]+last_box[3])/2
                speed = math.hypot(cx-lx, cy-ly)
                if last_area is not None and last_area > 0:
                    area_chg = abs(area - last_area) / (last_area + 1e-6)
            last_box, last_area = box, area

            if buf_feats:
                prev_speed = buf_feats[-1][0]
                acc = max(0.0, speed - prev_speed)

            features_vec = np.array([
                speed, acc, aspect, area/(W*H+1e-6), area_chg
            ], dtype=np.float32)
            buf_feats.append((speed, acc, aspect, area/(W*H+1e-6), area_chg))
            buf_times.append(t_sec)

            if len(buf_feats) >= window_frames:
                f = np.array(buf_feats[-window_frames:], dtype=np.float32)
                agg = np.concatenate([f.mean(axis=0), f.std(axis=0), f.max(axis=0)], axis=0)  # 15维

                t0, t1 = buf_times[-window_frames], buf_times[-1]
                rb_label, rb_conf = rule_behavior(
                    speed_px=float(f[:,0].mean()),
                    aspect_ratio=float(f[:,2].mean()),
                    area_change=float(f[:,4].mean())
                )
                mdl_pred = predict_with_model(agg)
                if mdl_pred is not None:
                    auto_label, auto_conf = mdl_pred
                    if auto_conf < 0.55 and rb_conf > auto_conf:
                        auto_label, auto_conf = rb_label, rb_conf
                else:
                    auto_label, auto_conf = rb_label, rb_conf

                bark, bark_s = bark_present(t0, t1)  # 恒为 False

                seg = Segment(
                    seg_id=str(uuid.uuid4()),
                    t_start=float(t0), t_end=float(t1),
                    features=agg, auto_label=auto_label, auto_conf=auto_conf,
                    bark=bool(bark)
                )
                segments.append(seg)

        progress.progress(min(1.0, (idx+step)/max(1,total_frames)))

    cap.release()

    if not segments:
        st.error("未检测到狗。请确保画面中有清晰的狗并有足够运动。")
        st.stop()

    st.subheader("分析结果（时间轴）")
    rows = []
    for s in segments:
        a,v,aff_c = affect_from_behavior(s.auto_label, s.bark)
        rows.append({
            "start(s)": round(s.t_start,2),
            "end(s)": round(s.t_end,2),
            "behavior": s.auto_label,
            "conf": round(s.auto_conf,2),
            "bark": "no",  # 当前禁用
            "arousal": round(a,2),
            "valence": round(v,2)
        })
    st.dataframe(rows, use_container_width=True)

    st.subheader("需要人工纠正的片段（低置信度）")
    n_flag = 0
    for s in segments:
        if s.auto_conf < lowconf_th:
            n_flag += 1
            with st.expander(f"{s.t_start:.2f}–{s.t_end:.2f}s  模型：{s.auto_label}（{s.auto_conf:.2f}）"):
                choice = st.selectbox("真实行为标签", LABELS, index=LABELS.index(s.auto_label),
                                      key=f"sel_{s.seg_id}")
                if st.button("保存为训练样本", key=f"save_{s.seg_id}"):
                    meta = {"t0": s.t_start, "t1": s.t_end, "bark": False}
                    save_sample(s.features, choice, meta)
                    st.success("样本已保存 ✅ 下次点击侧栏“改进模型”即可学习。")
    if n_flag == 0:
        st.caption("所有片段置信度都不错，无需标注。")

    if st.button("📄 导出本次报告(JSON)"):
        report = {
            "video": uploaded.name,
            "segments": [
                {
                    "t_start": s.t_start, "t_end": s.t_end,
                    "behavior": s.auto_label, "conf": s.auto_conf,
                    "bark": False,
                    "arousal": affect_from_behavior(s.auto_label, False)[0],
                    "valence": affect_from_behavior(s.auto_label, False)[1],
                } for s in segments
            ],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        rid = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORT_DIR, f"report_{rid}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        st.success(f"已导出：{path}")

st.markdown("---")
st.caption("当前为 Cloud 安全版（无音频）。当依赖安装稳定后，可恢复 librosa/scipy 并启用吠叫检测；或升级为 SLEAP/DeepLabCut 姿态 + VideoMAE/SlowFast。")
