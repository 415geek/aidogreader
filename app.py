# -*- coding: utf-8 -*-
# Dog Behavior & Affect Analyzer — Pro edition (Cloud-safe)
# - Lazy import Ultralytics; OpenCV headless safe
# - Simple/Pro sidebar; Evidence cards (head micro-expressions & tail)
# - Natural language summary + JSON & TXT report export
# - Optional online learning via sklearn (if installed)

import os, json, time, uuid, math, tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st

# -------------------- Safe OpenCV import --------------------
try:
    import cv2
except Exception as e:
    cv2 = None
    CV2_IMPORT_ERR = e
else:
    CV2_IMPORT_ERR = None

# -------------------- Optional sklearn --------------------
SK_OK = True
try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
except Exception:
    SK_OK = False

import joblib

# -------------------- App config --------------------
APP_TITLE = "🐶 Dog Behavior & Affect Analyzer"
DATA_DIR = "data_samples"; MODEL_DIR = "models"; REPORT_DIR = "reports"
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(REPORT_DIR, exist_ok=True)

LABELS = ["lying", "sitting/idle", "walking", "running", "sprinting/jumping"]
AFFECT_TABLE = {
    "lying": (0.20, 0.70),
    "sitting/idle": (0.30, 0.60),
    "walking": (0.45, 0.65),
    "running": (0.70, 0.65),
    "sprinting/jumping": (0.85, 0.60),
}

# -------------------- Dataclass --------------------
@dataclass
class Segment:
    seg_id: str
    t_start: float
    t_end: float
    features: np.ndarray
    auto_label: str
    auto_conf: float
    bark: bool  # audio disabled for cloud

# -------------------- Utils --------------------
def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0])*(a[3]-a[1]); areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

def rule_behavior(speed_px: float, aspect_ratio: float, area_change: float) -> Tuple[str, float]:
    if speed_px < 2.0:
        if aspect_ratio < 0.85 and area_change < 0.01: return "lying", 0.70
        return "sitting/idle", 0.60
    elif speed_px < 10.0: return "walking", 0.70
    elif speed_px < 23.0: return "running", 0.75
    else: return "sprinting/jumping", 0.80

def affect_from_behavior(label: str, bark: bool) -> Tuple[float, float, float]:
    a, v = AFFECT_TABLE.get(label, (0.5, 0.5))
    conf_aff = 0.45 if label in ["lying","sitting/idle"] else 0.55
    return a, v, conf_aff

# --- ROI & micro features ---
def crop_roi(frame, box, rel):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    rx1 = int(x1 + rel[0] * w); ry1 = int(y1 + rel[1] * h)
    rx2 = int(x1 + rel[2] * w); ry2 = int(y1 + rel[3] * h)
    rx1, ry1 = max(0, rx1), max(0, ry1)
    rx2, ry2 = min(frame.shape[1], rx2), min(frame.shape[0], ry2)
    if rx2 - rx1 < 4 or ry2 - ry1 < 4: return None
    return frame[ry1:ry2, rx1:rx2].copy()

def tail_wag_features(prev_tail_gray, tail_gray):
    """尾巴 ROI 光流近似特征（增加防错机制）"""
    # 检查输入有效性
    if prev_tail_gray is None or tail_gray is None:
        return None
    if not isinstance(prev_tail_gray, np.ndarray) or not isinstance(tail_gray, np.ndarray):
        return None
    if prev_tail_gray.shape != tail_gray.shape:
        try:
            tail_gray = cv2.resize(tail_gray, (prev_tail_gray.shape[1], prev_tail_gray.shape[0]))
        except Exception:
            return None
    try:
        diff = cv2.absdiff(tail_gray, prev_tail_gray)
    except Exception:
        return None

    mag = float(np.mean(diff))  # 摆动强度近似
    gx = cv2.Sobel(tail_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(tail_gray, cv2.CV_32F, 0, 1, ksize=3)
    ori_ratio = float(np.mean(np.abs(gx))) / (np.mean(np.abs(gy)) + 1e-6)
    return {"wag_mag": mag, "wag_orient": ori_ratio}
def head_micro_features(head_bgr):
    if head_bgr is None: return None
    img = cv2.resize(head_bgr, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    eye_roi = blur[20:64, 32:96]
    mouth_roi = blur[72:120, 24:104]
    ear_left  = blur[8:40,  0:40]
    ear_right = blur[8:40, 88:128]
    _, eye_th = cv2.threshold(eye_roi, 0, 255, cv2.THRESH_OTSU)
    eye_open = 1.0 - (np.mean(eye_th)/255.0)
    edges = cv2.Canny(mouth_roi, 50, 120)
    mouth_open = float(np.mean(edges > 0))
    ear_l_edge = float(np.mean(cv2.Canny(ear_left, 50, 120) > 0))
    ear_r_edge = float(np.mean(cv2.Canny(ear_right, 50, 120) > 0))
    ear_up = (ear_l_edge + ear_r_edge) / 2.0
    return {"eye_open": eye_open, "mouth_open": mouth_open, "ear_up": ear_up}

# -------------------- Samples I/O --------------------
def save_sample(features: np.ndarray, true_label: str, meta: dict):
    sid = str(uuid.uuid4())
    np.save(os.path.join(DATA_DIR, f"{sid}_x.npy"), features.astype(np.float32))
    with open(os.path.join(DATA_DIR, f"{sid}_y.json"), "w") as f:
        json.dump({"y": true_label, "meta": meta}, f)

def load_samples(limit: Optional[int] = None):
    files = [f for f in os.listdir(DATA_DIR) if f.endswith("_y.json")]
    if not files: return None, None
    if limit: files = files[:limit]
    Xs, ys = [], []
    for jf in files:
        meta = json.load(open(os.path.join(DATA_DIR, jf)))
        y = meta["y"]; sid = jf.replace("_y.json", "")
        x = np.load(os.path.join(DATA_DIR, f"{sid}_x.npy"))
        Xs.append(x); ys.append(LABELS.index(y))
    return np.vstack(Xs), np.array(ys, dtype=np.int64)

def fit_or_partial_update(X_train: np.ndarray, y_train: np.ndarray):
    if not SK_OK: return None, None, None
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train)
    Xs = scaler.transform(X_train)
    base = SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="optimal", random_state=42)
    classes = np.arange(len(LABELS)); base.partial_fit(Xs, y_train, classes=classes)
    clf = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3).fit(Xs, y_train)
    ts = time.strftime("%Y%m%d_%H%M%S")
    joblib.dump(clf, os.path.join(MODEL_DIR, f"behavior_clf_{ts}.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{ts}.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "behavior_clf_latest.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_latest.joblib"))
    return clf, scaler, ts

def predict_with_model(features_vec: np.ndarray):
    if not SK_OK: return None
    mp, sp = os.path.join(MODEL_DIR, "behavior_clf_latest.joblib"), os.path.join(MODEL_DIR, "scaler_latest.joblib")
    if not (os.path.exists(mp) and os.path.exists(sp)): return None
    clf, scaler = joblib.load(mp), joblib.load(sp)
    probs = clf.predict_proba(scaler.transform(features_vec.reshape(1,-1)))[0]
    idx = int(np.argmax(probs)); return LABELS[idx], float(probs[idx])

# -------------------- Lazy import YOLO --------------------
@st.cache_resource
def load_detector():
    try:
        from ultralytics import YOLO
    except Exception as e:
        st.error(
            "Ultralytics 导入失败。请确保 requirements.txt 包含：\n"
            "opencv-python-headless==4.8.1.78, ultralytics==8.2.103, numpy==1.26.4\n"
            "并 Clear cache 后 Reboot。\n"
            f"原始异常：{repr(e)}"
        ); st.stop()
    return YOLO("yolov8n.pt")

# -------------------- Summary & explanations --------------------
def interpret_affective_state(segments):
    mean_arousal = np.mean([affect_from_behavior(s.auto_label, s.bark)[0] for s in segments])
    mean_valence = np.mean([affect_from_behavior(s.auto_label, s.bark)[1] for s in segments])
    behavior_counts = {}
    for s in segments: behavior_counts[s.auto_label] = behavior_counts.get(s.auto_label, 0) + 1
    main_behavior = max(behavior_counts, key=behavior_counts.get)
    ar, va = round(mean_arousal,2), round(mean_valence,2)

    if ar > 0.7 and va > 0.6:
        mood, reason, advice = "非常兴奋且愉快", "常见于玩耍/奔跑或与熟人互动。", "此时互动或训练效率最高。"
    elif ar < 0.4 and va > 0.6:
        mood, reason, advice = "平静且安全", "环境可预期、无压力刺激。", "保持安稳环境与轻柔抚触。"
    elif ar > 0.7 and va < 0.5:
        mood, reason, advice = "紧张或过度兴奋", "对环境过度反应，或存在轻度焦虑。", "用低强度游戏转移注意，避免突发刺激。"
    elif ar < 0.4 and va < 0.5:
        mood, reason, advice = "情绪低落或疲惫", "活动减少与长时间卧躺。", "关注饮食睡眠，必要时增加轻度外出或体检。"
    else:
        mood, reason, advice = "中性平稳", "休息与活动交替的正常状态。", "保持当前作息即可。"

    return (
        f"🐕 这段视频中以 **{main_behavior}** 为主；整体情绪 **{mood}**。\n"
        f"{reason}\n"
        f"科学指标：唤醒度 {ar:.2f}、效价 {va:.2f}。\n"
        f"建议：{advice}"
    )

def explain_micro(ev: dict):
    out = []
    if not ev: return out
    if ev.get("wag_mag_mu",0) > 6.0: out.append("尾摆幅度明显 → 唤醒度较高（兴奋/紧张可能）。")
    if ev.get("ear_up_mu",0) > 0.20: out.append("耳位上扬 → 警觉/关注度高。")
    if ev.get("ear_up_mu",1) < 0.10: out.append("耳位放松/后贴 → 安全感较高。")
    if ev.get("eye_open_mu",1) < 0.28: out.append("眼裂较小 → 放松或轻度疲惫。")
    if ev.get("mouth_open_mu",0) > 0.18: out.append("口裂边缘活跃 → 喘气/缓解压力行为。")
    return out

# -------------------- Page --------------------
st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)

if cv2 is None:
    st.error("OpenCV 未正确加载。请使用 opencv-python-headless==4.8.1.78，并 Clear cache 后 Reboot。")
    if CV2_IMPORT_ERR: st.caption(f"导入异常：{repr(CV2_IMPORT_ERR)}")
    st.stop()

with st.sidebar:
    st.header("设置")
    pro_mode = st.toggle("高级模式（面向专业用户）", value=False)
    PRESETS = {
        "家庭室内（普通）":       {"conf_th": 0.35, "sample_fps": 6,  "max_seconds": 25},
        "家庭院子/户外（光线足）": {"conf_th": 0.30, "sample_fps": 6,  "max_seconds": 25},
        "弱光/模糊（更稳）":       {"conf_th": 0.45, "sample_fps": 5,  "max_seconds": 30},
        "运动多（更快）":         {"conf_th": 0.35, "sample_fps": 8,  "max_seconds": 20},
    }
    if not pro_mode:
        preset = st.selectbox("场景预设", list(PRESETS.keys()), index=0)
        speed_vs_acc = st.slider("速度 ↔ 准确度", 0, 10, 6)
        base = PRESETS[preset]
        conf_th   = float(np.clip(base["conf_th"] + (5 - speed_vs_acc) * 0.01, 0.20, 0.55))
        sample_fps = int(np.clip(base["sample_fps"] + (speed_vs_acc - 5) * 0.5, 3, 12))
        max_seconds = int(np.clip(base["max_seconds"] + (5 - speed_vs_acc) * 1.5, 10, 60))
        lowconf_th = 0.65
        st.caption(f"策略：阈值≈{conf_th:.2f}，抽帧≈{sample_fps} fps，最长 {max_seconds}s。")
    else:
        max_seconds = st.slider("分析时长上限(秒)", 5, 120, 25)
        conf_th = st.slider("检测置信度阈值（YOLO）", 0.1, 0.9, 0.35)
        sample_fps = st.slider("分析抽帧速率(fps)", 3, 24, 6)
        lowconf_th = st.slider("低置信度阈值（触发标注）", 0.50, 0.90, 0.65)
        if SK_OK:
            st.markdown("---")
            if st.button("🧠 使用已标注样本改进模型"):
                X_all, y_all = load_samples()
                if X_all is None: st.warning("暂无标注样本。先在下方时间轴中保存几条训练样本。")
                else:
                    _, _, tag = fit_or_partial_update(X_all, y_all)
                    st.success(f"模型已更新 ✅（版本 {tag}）")
        else:
            st.info("增量学习未启用（scikit-learn 未安装）。")

st.caption("上传短视频，进行行为与情绪（唤醒/效价）推断；并输出“证据卡+自然语言总结”。")
uploaded = st.file_uploader("上传视频 (mp4/mov/mkv)", type=["mp4","mov","mkv"])

if uploaded:
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmpf.write(uploaded.read()); tmpf.close()

    det = load_detector()
    cap = cv2.VideoCapture(tmpf.name)

    raw_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if raw_fps < 20: sample_fps = min(sample_fps, 6)

    fps = raw_fps or 30
    total_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps*max_seconds, max_seconds*fps))
    step = max(1, int(round(fps / sample_fps)))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.info("开始分析（抽帧提速）。")
    progress = st.progress(0.0)

    last_box = None; last_area = None
    segments: List[Segment] = []
    window_frames = max(3, int(sample_fps * 1.2))  # ~1.2s
    buf_feats, buf_times = [], []
    prev_tail_gray = None
    micro_buf = []

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

            features_vec = np.array([speed, acc, aspect, area/(W*H+1e-6), area_chg], dtype=np.float32)
            buf_feats.append((speed, acc, aspect, area/(W*H+1e-6), area_chg))
            buf_times.append(t_sec)

            # --- micro features (head & tail) ---
            head_roi = crop_roi(frame, box, (0.15, 0.00, 0.85, 0.45))
            tail_roi = crop_roi(frame, box, (0.55, 0.60, 1.00, 1.00))
            tail_gray = cv2.cvtColor(tail_roi, cv2.COLOR_BGR2GRAY) if tail_roi is not None else None
            tail_feats = tail_wag_features(prev_tail_gray, tail_gray) if tail_gray is not None else None
            prev_tail_gray = tail_gray
            head_feats = head_micro_features(head_roi)
            micro_buf.append({"t": t_sec, "head": head_feats, "tail": tail_feats})

            # --- segment assemble ---
            if len(buf_feats) >= window_frames:
                f = np.array(buf_feats[-window_frames:], dtype=np.float32)
                agg = np.concatenate([f.mean(axis=0), f.std(axis=0), f.max(axis=0)], axis=0)  # 15 dims
                t0, t1 = buf_times[-window_frames], buf_times[-1]

                rb_label, rb_conf = rule_behavior(float(f[:,0].mean()), float(f[:,2].mean()), float(f[:,4].mean()))
                mdl_pred = predict_with_model(agg) if SK_OK else None
                if mdl_pred is not None:
                    auto_label, auto_conf = mdl_pred
                    if auto_conf < 0.55 and rb_conf > auto_conf:
                        auto_label, auto_conf = rb_label, rb_conf
                else:
                    auto_label, auto_conf = rb_label, rb_conf

                seg = Segment(
                    seg_id=str(uuid.uuid4()),
                    t_start=float(t0), t_end=float(t1),
                    features=agg, auto_label=auto_label, auto_conf=auto_conf,
                    bark=False
                )

                # attach micro-evidence
                win_samples = [m for m in micro_buf if m["t"] >= t0-1e-3 and m["t"] <= t1+1e-3]
                def agg_stats(key):
                    vals = [s["head"][key] for s in win_samples if s["head"] and s["head"].get(key) is not None]
                    return (float(np.mean(vals)) if vals else None, float(np.std(vals)) if vals else None)
                eye_mu,_ = agg_stats("eye_open"); mouth_mu,_ = agg_stats("mouth_open"); ear_mu,_ = agg_stats("ear_up")
                wag_mag = [s["tail"]["wag_mag"] for s in win_samples if s["tail"] and s["tail"].get("wag_mag") is not None]
                wag_or  = [s["tail"]["wag_orient"] for s in win_samples if s["tail"] and s["tail"].get("wag_orient") is not None]
                seg.micro = {
                    "eye_open_mu": (None if eye_mu is None else float(eye_mu)),
                    "mouth_open_mu": (None if mouth_mu is None else float(mouth_mu)),
                    "ear_up_mu": (None if ear_mu is None else float(ear_mu)),
                    "wag_mag_mu": (None if not wag_mag else float(np.mean(wag_mag))),
                    "wag_orient_mu": (None if not wag_or else float(np.mean(wag_or))),
                }
                segments.append(seg)

        progress.progress(min(1.0, (idx+step)/max(1,total_frames)))

    cap.release()

    if not segments:
        st.error("未检测到狗。请确保画面中有清晰的狗并有足够运动。")
        st.stop()

    # ---------- Table ----------
    st.subheader("分析结果（时间轴）")
    rows = []
    for s in segments:
        a,v,_ = affect_from_behavior(s.auto_label, s.bark)
        rows.append({
            "start(s)": round(s.t_start,2), "end(s)": round(s.t_end,2),
            "behavior": s.auto_label, "conf": round(s.auto_conf,2),
            "arousal": round(a,2), "valence": round(v,2)
        })
    st.dataframe(rows, use_container_width=True)

    # ---------- Evidence cards ----------
    st.subheader("🧠 微表情与尾部证据卡（自动解释）")
    for s in segments:
        ev = getattr(s, "micro", None)
        if not ev: continue
        explain = explain_micro(ev) or ["未发现显著微表情信号，整体处于中性范围。"]
        with st.expander(f"{s.t_start:.2f}–{s.t_end:.2f}s  证据与解释（{s.auto_label}，{s.auto_conf:.2f}）"):
            st.write("\n".join(f"- {t}" for t in explain))
            st.json(ev)

    # ---------- NL summary ----------
    st.subheader("🧩 行为心理总结")
    summary_text = interpret_affective_state(segments)
    st.info(summary_text)

    # ---------- Reports ----------
    if st.button("📄 导出本次报告(JSON+TXT)"):
        rid = time.strftime("%Y%m%d_%H%M%S")
        report = {
            "video": uploaded.name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary_text": summary_text,
            "segments_detailed": [
                {
                    "t_start": s.t_start, "t_end": s.t_end,
                    "behavior": s.auto_label, "conf": s.auto_conf,
                    "affect": {
                        "arousal": affect_from_behavior(s.auto_label, False)[0],
                        "valence": affect_from_behavior(s.auto_label, False)[1]
                    },
                    "micro_evidence": getattr(s, "micro", None),
                    "micro_explain": explain_micro(getattr(s, "micro", None)),
                } for s in segments
            ]
        }
        json_path = os.path.join(REPORT_DIR, f"report_{rid}.json")
        with open(json_path, "w", encoding="utf-8") as f: json.dump(report, f, indent=2)

        txt_lines = [summary_text, "", "—— 细节证据 ——"]
        for sd in report["segments_detailed"]:
            txt_lines.append(f"[{sd['t_start']:.1f}-{sd['t_end']:.1f}s] {sd['behavior']} (conf={sd['conf']:.2f})")
            if sd["micro_explain"]:
                for t in sd["micro_explain"]: txt_lines.append(f"  · {t}")
            else:
                txt_lines.append("  · 无显著微表情信号")
        txt_path = os.path.join(REPORT_DIR, f"report_{rid}.txt")
        with open(txt_path, "w", encoding="utf-8") as f: f.write("\n".join(txt_lines))

        st.success(f"已导出：{json_path}  和  {txt_path}")

st.markdown("---")
st.caption("Cloud 安全版：OpenCV headless，音频关闭；安装 scikit-learn 后自动启用增量学习按钮。")
