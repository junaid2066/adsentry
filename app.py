# --- Full AdSentry Streamlit App with Tabs ---
import os
import cv2
import torch
import streamlit as st
import numpy as np
import pandas as pd
from transformers import ViTImageProcessor, ViTModel
from decord import VideoReader, cpu
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

# --- Streamlit config
st.set_page_config(page_title="Ad Detector with ViT", layout="wide")
st.title("🎥 AdSentry: Smart Video Advertisement Detection & Counting System")

# --- AITeC Branding CSS Styling ---
st.markdown("""
<style>
/* Custom styles for branding and layout */
    .stApp {
        background-color: #ffffff;
    }
    
    header .stMarkdown h1 { 
        color: #1F2B14; 
        font-weight: bold; 
    }

    section[data-testid="stSidebar"] > div:first-child {
        background-color: #1F2B14;
    }

    section[data-testid="stSidebar"] > div:first-child {
        background-color: #1F2B14;
    }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #FFD700 !important;
    }

    .ad-stats {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 20px;
    }

    .ad-stats div {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 80px;
        padding: 12px 24px;
        border-radius: 10px;
        color: white;
        font-size: 18px;
        font-weight: bold;
        white-space: nowrap;
        overflow: hidden;
    }


.ad-count { background-color: #FFD700; color: #1F2B14; }

.ad-duration { background-color: #1F2B14; color: #FFD700; }


div.stButton > button:first-child {
    background-color: #FFD700; color: #1F2B14;
    height: 3em; width: 70%; border-radius: 10px; border: none; font-weight: bold;
}
div.stButton > button:first-child:hover { background-color: #e6c200; color: white; }



.footer-bar {
    position: fixed; bottom: 0; left: 0; width: 100%;
    background-color: #1F2B14; border-top: 1px solid #ccc;
    padding: 12px; font-size: 14px; color: #FFD700; z-index: 9999;
    display: flex; justify-content: center; align-items: center;
    box-shadow: 0 -1px 5px rgba(0,0,0,0.1);
}
.footer-bar .footer-content { text-align: center; line-height: 1.5; }


</style>
""", unsafe_allow_html=True)

# --- Load model
@st.cache_resource
def load_model():
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    return model.eval(), processor

model, processor = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --- Extract frames
def extract_frames(video_path, sample_rate=15):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = list(range(0, total_frames, sample_rate))
    frames = vr.get_batch(indices).asnumpy()
    return frames

# --- Get embeddings
def get_frame_embeddings(frames):
    embeddings = []
    for frame in frames:
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(emb[0])
    return np.array(embeddings)

# --- Detect ads
def detect_ad(main_embs, ad_embs, threshold=0.90):
    matches = []
    ad_len = len(ad_embs)
    for i in range(len(main_embs) - ad_len + 1):
        chunk = main_embs[i:i + ad_len]
        sim = cosine_similarity(chunk, ad_embs).diagonal().mean()
        if sim > threshold:
            matches.append((i, sim))
    return matches

# --- Merge close detections
def merge_detections(detections, sample_rate, fps=25, min_gap_sec=5):
    merged = []
    last_time = -1
    for idx, sim in detections:
        current_sec = idx * sample_rate / fps
        if last_time < 0 or (current_sec - last_time) > min_gap_sec:
            merged.append((current_sec, sim))
            last_time = current_sec
    return merged

# --- Sidebar UI
st.sidebar.header("📁 Upload Videos")
main_file = st.sidebar.file_uploader("Main Video", type=["mp4", "avi"])
ad_file = st.sidebar.file_uploader("Ad Segment", type=["mp4", "avi"])
sample_rate = st.sidebar.slider("Frame Sampling Rate", 5, 60, 15)
similarity_thresh = st.sidebar.slider("Cosine Similarity Threshold", 0.80, 0.99, 0.90, 0.01)
min_gap = st.sidebar.slider("Duration of Advertisement Video (sec)", 1, 60, 5)
start_detection = st.sidebar.button("🚀 Start Detection")

# --- Tabs Layout ---
tab1, tab2, tab3 = st.tabs(["📖 Overview", "📊 Analytics", "📄 Report"])

# --- Tab 1: Overview
with tab1:
    st.subheader("🔹 What is it?")
    st.markdown("""
**AdSentry** is an AI-powered application that automatically detects and counts repeated **advertisement segments** within longer video content.  
By leveraging advanced machine learning models and visual similarity analysis, AdSentry enables broadcasters, media analysts, and compliance teams to quickly identify when, where, and how often specific ad segments appear in a video.
""")

    st.subheader("🔹 How to Use It")
    st.markdown("""
1. **Upload Videos:**
   - Upload the **main video** (e.g., a broadcast, show, or long-form content).
   - Upload the **ad sample** video (the specific advertisement you want to search for).

2. **Set Parameters:**
   - Choose a **frame sampling rate** (how frequently frames are extracted for analysis). **Ideal Frame rate = 15fps**.
   - Set a **cosine similarity threshold** to control match strictness. **Ideal Cosine Similarity = 0.8**.
   - Define a **minimum gap** (in seconds) to merge nearby detections into one ad instance.

3. **Run Detection:**
   - Click **“Start Detection”** to begin.
   - The app analyzes and compares frames, detecting similar visual patterns between the main content and the ad sample.

4. **Review Results:**
   - View the **number of ad matches** and **total ad duration**.
   - Go to the **“Report”** tab to **download a CSV** file for external use or compliance tracking.
""")

    st.subheader("🔹 Key Features")
    st.markdown("""
- ✅ Upload and analyze any main video and ad sample  
- ✅ Accurate ad segment detection using state-of-the-art visual models  
- ✅ Customizable detection sensitivity and sampling control  
- ✅ Visual summary of detected ad count and total duration  
- ✅ Downloadable CSV report of ad detections  
- ✅ User-friendly Streamlit interface with custom branding
""")

    st.subheader("🔹 Use Cases")
    st.markdown("""
- Auditing TV broadcasts for ad placements  
- Ensuring ad compliance for regulatory reporting  
- Detecting repeated or unauthorized advertisements  
- Researching advertising frequency in news/media channels  
- Automating ad monitoring for OTT and digital media platforms
""")

# --- Tab 2: Analytics
with tab2:
    if start_detection:
        if not main_file or not ad_file:
            st.error("Please upload both the main video and ad segment.")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                main_path = os.path.join(temp_dir, "main.mp4")
                ad_path = os.path.join(temp_dir, "ad.mp4")

                with open(main_path, "wb") as f:
                    f.write(main_file.read())
                with open(ad_path, "wb") as f:
                    f.write(ad_file.read())

                st.info("📦 Extracting frames from both videos...")
                main_frames = extract_frames(main_path, sample_rate)
                ad_frames = extract_frames(ad_path, sample_rate)

                st.write(f"🖼️ Extracted {len(main_frames)} frames from main video.")
                st.write(f"🖼️ Extracted {len(ad_frames)} frames from ad segment.")

                st.info("🔬 Extracting visual embeddings...")
                main_embs = get_frame_embeddings(main_frames)
                ad_embs = get_frame_embeddings(ad_frames)

                st.info("🔎 Comparing embeddings with cosine similarity...")
                detections = detect_ad(main_embs, ad_embs, threshold=similarity_thresh)

                if not detections:
                    st.warning("❌ No matching ad segments found. Try lowering the similarity threshold.")
                    st.session_state.csv_data = None
                else:
                    filtered = merge_detections(detections, sample_rate, fps=25, min_gap_sec=min_gap)

                    ad_duration_sec = (len(ad_frames) * sample_rate) / 25
                    total_ad_time = len(filtered) * ad_duration_sec
                    minutes = int(total_ad_time // 60)
                    seconds = int(total_ad_time % 60)
                    duration_msg = f"{total_ad_time:.2f} seconds ({minutes} min {seconds} sec)"

                    st.markdown(f"""
                    <div class="ad-stats">
                        <div class="ad-count">🔢 {len(filtered)} Ad Matches Detected</div>
                        <div class="ad-duration">🕒 Total Ad Duration: {duration_msg}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    for sec, sim in filtered:
                        st.write(f"📍 At ~{sec:.2f} seconds, similarity: {sim:.4f}")

                    st.session_state.csv_data = pd.DataFrame({
                        "Ad Start Time (sec)": [round(sec, 2) for sec, _ in filtered],
                        "Cosine Similarity": [round(sim, 4) for _, sim in filtered]
                    })

# --- Tab 3: Report
with tab3:
    if "csv_data" in st.session_state and st.session_state.csv_data is not None:
        st.dataframe(st.session_state.csv_data)
        st.download_button(
            label="📥 Download CSV Report",
            data=st.session_state.csv_data.to_csv(index=False).encode("utf-8"),
            file_name="ad_matches.csv",
            mime="text/csv"
        )
    else:
        st.info("⚠️ No detection report available. Run analysis in the Analytics tab.")

# --- Footer
st.markdown("""
<div class="footer-bar">
    <div class="footer-content">
        <strong>Designed and Developed by Natural Language Processing Lab</strong><br>
        Artificial Intelligence Technology Centre (AITeC) | National Centre for Physics (NCP)
    </div>
</div>
""", unsafe_allow_html=True)
