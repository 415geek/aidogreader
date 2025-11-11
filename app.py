# -*- coding: utf-8 -*-
# app.py — Dog Behavior & Affect Analyzer (Production-Ready Skeleton)
# 功能：上传视频 -> 检测追踪 -> 特征提取 -> 行为识别(可增量学习) -> 情绪映射 -> 时间轴报告/可视化
# 依赖：streamlit, ultralytics, opencv-python, librosa, soundfile, scikit-learn, joblib, numpy

import os, io, json, time, uuid, math, tempfile
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import streamlit as st
import cv2
from ultralytics import YOLO
import librosa
import soundfile as sf
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
import joblib

# ------------ 全局配置 ------------
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
CLASS_COLORS = {
    "lying": (120, 200, 80),
    "sitting/idle": (200, 200, 80),
    "walking": (80, 180, 220),
    "running": (80, 120, 240),
    "sprinting/jumping": (60, 60, 255),
}

# ------------ 数据结构 ------------
@dataclass
class Segment:
    seg_id: str
    t_start: float
    t_end: float
    features: np.ndarray
    auto_label: str
    auto_conf: float
    bark: bool

# ------------ 模型加载 ------------
@st.cache_resource
def load_detector():
    # 轻量模型即可，首次自动下载
    return YOLO("yolov8n.pt")

@st.cache_resource
def init_or_load_clf():
    model_p = os.path.join(MODEL_DIR, "behavior_clf_latest.joblib")
    scaler_p = os.path.join(MODEL_DIR, "scaler_latest.joblib")
    if os.path.exists(model_p) and os.path.exists(scaler_p):
        clf = joblib.load(model_p)
        scaler = joblib.load(scaler_p)
    else:
        # 初始化（先用规则生成的“伪标签”做冷启，后续通过主动学习增量修正）
        base = SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="optimal", random_state=42)
        clf = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
        scaler = StandardScaler(with_mean=True, with_std=True)
        # 尚未拟合，但先存“空壳”，避免首次调用报错
        joblib.dump(clf, model_p); joblib.dump(scaler, scaler_p)
    return clf, scaler

# ------------ 工具函数 ------------
def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0])*(a[3]-a[1]); areaB = (b[2]-b[0])*(b[3]-b[1])
    union = areaA + areaB - inter + 1e-6
    return inter / union

def extract_audio_signal(video_path, target_sr=16000):
    try:
        y, sr = librosa.load(video_path, sr=target_sr, mono=True)
        return y, sr
    except Exception:
        return None, None

def bark_score_track(y, sr, frame_ms=400, hop_ms=160) -> List[Tuple[float, float, float]]:
    """返回[(start_s, end_s, bark_score)]。bark_score=能量*高频占比的规范化分数。"""
    if y is None: return []
    frame_len = int(frame_ms/1000*sr); hop_len = int(hop_ms/1000*sr)
    out = []
    i = 0
    # 动态阈值：基于整段 RMS
    rms = np.sqrt(np.mean(y**2) + 1e-9)
    base_e = max(1e-6, rms**2)
    while i + frame_len <= len(y):
        seg = y[i:i+frame_len]
        energy = float(np.mean(seg**2) / base_e)  # 相对能量
        S = np.abs(librosa.stft(seg, n_fft=512, hop_length=128))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
        mask = (freqs >= 700) & (freqs <= 3500)  # 犬吠主能带（粗略）
        ratio = (np.sum(S[mask]) + 1e-6) / (np.sum(S) + 1e-6)
        score = float(energy * ratio)
        out.append((i/sr, (i+frame_len)/sr, score))
        i += hop_len
    return out

def rule_behavior(speed_px, aspect_ratio, area_change):
    # 速度主导，形态修正（躺下时长宽比偏小/面积波动小）
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
    if bark: a = min(1.0, a + 0.1)
    conf_aff = 0.45 if label in ["lying","sitting/idle"] else 0.55
    return a, v, conf_aff

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
    # 全量小训练（更稳），你也可以把 base_clf.partial_fit 做成真·在线
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_train)
    Xs = scaler.transform(X_train)
    base = SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="optimal", random_state=42)
    classes = np.arange(len(LABELS))
    base.partial_fit(Xs, y_train, classes=classes)
    calibrated = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
    calibrated.fit(Xs, y_train)
    ts = time.strftime("%Y%m%d_%H%M%S")
    joblib.dump(calibrated, os.path.join(MODEL_DIR, f"behavior_clf_{ts}.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{ts}.joblib"))
    joblib.dump(calibrated, os.path.join(MODEL_DIR, "behavior_clf_latest.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_latest.joblib"))
    return calibrated, scaler, ts

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

# ------------ Streamlit UI ------------
st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)
st.caption("上传一段短视频，系统将输出：行为识别、置信度、吠叫概率以及情绪（唤醒/效价）推断，并支持边用边学。")

with st.sidebar:
    st.header("参数")
    max_seconds = st.slider("分析时长上限(秒)", 5, 90, 25)
    conf_th = st.slider("检测置信度阈值（YOLO）", 0.1, 0.8, 0.35)
    sample_fps = st.slider("分析抽帧速率(fps)", 3, 12, 6)
    bark_th = st.slider("吠叫分数阈值", 0.35, 2.0, 0.65)
    lowconf_th = st.slider("低置信度阈值（触发标注）", 0.50, 0.90, 0.65)
    st.markdown("---")
    do_train = st.button("🧠 使用已标注样本改进模型")

uploaded = st.file_uploader("上传视频 (mp4/mov/mkv)", type=["mp4","mov","mkv"])

if do_train:
    X_all, y_all = load_samples()
    if X_all is None:
        st.warning("暂无标注样本。先在下方时间轴中保存几条训练样本。")
    else:
        _, _, tag = fit_or_partial_update(X_all, y_all)
        st.success(f"模型已更新 ✅（版本 {tag) }）")

if uploaded:
    # 缓存临时视频
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmpf.write(uploaded.read()); tmpf.close()

    # 音频 → 吠叫分数轨迹
    audio, sr = extract_audio_signal(tmpf.name)
    bark_track = bark_score_track(audio, sr) if sr else []
    def bark_present(t0, t1):
        if not bark_track: return False, 0.0
        scores = [s for (a,b,s) in bark_track if not (b <= t0 or a >= t1)]
        if not scores: return False, 0.0
        smax = float(np.max(scores))
        return (smax >= bark_th), smax

    # 加载检测器
    det = load_detector()
    cap = cv2.VideoCapture(tmpf.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), max_seconds*fps))
    step = max(1, int(round(fps / sample_fps)))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.info("开始分析：为提速将抽帧处理（不影响最终判断的稳定性）。")
    progress = st.progress(0)
    last_box = None
    last_area = None
    segments: List[Segment] = []

    # 时窗聚合（按 N 帧为一小段）
    window_frames = max(3, int(sample_fps * 1.2))  # ~1.2s
    buf_feats, buf_times = [], []

    for idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        t_sec = idx / fps

        # YOLO 推理
        res = det(frame, conf=conf_th, verbose=False)[0]
        dog_boxes = []
        for b in res.boxes:
            cls = int(b.cls[0].item())
            if det.model.names.get(cls, "") == "dog":
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                dog_boxes.append([x1,y1,x2,y2, float(b.conf[0].item())])

        if dog_boxes:
            # 取最大框（多犬场景默认主角），亦可改成多目标
            dog_boxes.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
            box = dog_boxes[0][:4]

            # 速度/加速度/形态特征
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            w_box, h_box = box[2]-box[0], box[3]-box[1]
            area = w_box * h_box
            aspect = min(w_box, h_box) / (max(w_box, h_box)+1e-6)

            speed = 0.0; acc = 0.0; area_chg = 0.0
            if last_box is not None and iou(last_box, box) > 0.1:
                lx, ly = (last_box[0]+last_box[2])/2, (last_box[1]+last_box[3])/2
                speed = math.hypot(cx-lx, cy-ly)  # 像素/步
                if last_area is not None and last_area > 0:
                    area_chg = abs(area - last_area) / (last_area + 1e-6)
            last_box, last_area = box, area

            # 简单一阶差分得到“加速度”的代理（用速度与上一次速度的差）
            if buf_feats:
                prev_speed = buf_feats[-1][0]
                acc = max(0.0, speed - prev_speed)

            # 汇入缓冲
            features_vec = np.array([
                speed, acc, aspect, area/(W*H+1e-6), area_chg
            ], dtype=np.float32)
            buf_feats.append((speed, acc, aspect, area/(W*H+1e-6), area_chg))
            buf_times.append(t_sec)

            # 窗口聚合成片段
            if len(buf_feats) >= window_frames:
                f = np.array(buf_feats[-window_frames:], dtype=np.float32)
                # 统计特征（均值、方差、最大值）
                agg = np.concatenate([
                    f.mean(axis=0),
                    f.std(axis=0),
                    f.max(axis=0),
                ], axis=0)  # 5维*3=15维
                t0, t1 = buf_times[-window_frames], buf_times[-1]

                # 规则预测（作为冷启与备选）
                rb_label, rb_conf = rule_behavior(
                    speed_px=float(f[:,0].mean()),
                    aspect_ratio=float(f[:,2].mean()),
                    area_change=float(f[:,4].mean())
                )

                # 监督模型预测（若已有）
                mdl_pred = predict_with_model(agg)
                if mdl_pred is not None:
                    auto_label, auto_conf = mdl_pred
                    # 若模型置信低于规则，采用规则结果增稳
                    if auto_conf < 0.55 and rb_conf > auto_conf:
                        auto_label, auto_conf = rb_label, rb_conf
                else:
                    auto_label, auto_conf = rb_label, rb_conf

                # 吠叫判断
                bark, bark_s = bark_present(t0, t1)

                seg = Segment(
                    seg_id=str(uuid.uuid4()),
                    t_start=float(t0), t_end=float(t1),
                    features=agg, auto_label=auto_label, auto_conf=auto_conf,
                    bark=bool(bark)
                )
                segments.append(seg)

        progress.progress(min(1.0, (idx+step)/max(1,total_frames)))

    cap.release()

    # ------- 报告与标注 UI -------
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
            "bark": "yes" if s.bark else "no",
            "arousal": round(a,2),
            "valence": round(v,2)
        })
    st.dataframe(rows, use_container_width=True)

    # 低置信度 → 主动学习标注
    st.subheader("需要你来教一教（低置信度片段）")
    n_flag = 0
    for s in segments:
        if s.auto_conf < lowconf_th:
            n_flag += 1
            with st.expander(f"{s.t_start:.2f}–{s.t_end:.2f}s  模型：{s.auto_label}（{s.auto_conf:.2f}） | 吠叫={'是' if s.bark else '否'}"):
                choice = st.selectbox("真实行为标签", LABELS, index=LABELS.index(s.auto_label),
                                      key=f"sel_{s.seg_id}")
                if st.button("保存为训练样本", key=f"save_{s.seg_id}"):
                    meta = {"t0": s.t_start, "t1": s.t_end, "bark": s.bark}
                    save_sample(s.features, choice, meta)
                    st.success("样本已保存 ✅ 下次点击侧栏“改进模型”即可学习。")
    if n_flag == 0:
        st.caption("所有片段置信度都不错，无需标注。")

    # 快速报告导出（JSON）
    if st.button("📄 导出本次报告(JSON)"):
        report = {
            "video": uploaded.name,
            "segments": [
                {
                    "t_start": s.t_start, "t_end": s.t_end,
                    "behavior": s.auto_label, "conf": s.auto_conf,
                    "bark": s.bark,
                    "arousal": affect_from_behavior(s.auto_label, s.bark)[0],
                    "valence": affect_from_behavior(s.auto_label, s.bark)[1],
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
    st.caption("提示：要显著提升准确率，请累积你的真实场景标注样本，然后在侧栏执行“改进模型”。也可以替换 YOLO 权重为你的自训模型，或接入动物姿态估计（DLC/SLEAP）进一步细化“坐/趴/摇尾/抓挠/舔爪”等原子行为。")
