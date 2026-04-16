## 🎥 AdSentry: Smart Video Advertisement Detection & Counting System
AdSentry is an AI-powered video analytics system designed to automatically detect, count, and analyze repeated advertisement segments within video content.
It leverages Vision Transformers (ViT) and cosine similarity-based matching to identify advertisement occurrences with high accuracy.

This system is particularly useful for media monitoring, marketing analytics, and broadcast compliance.

----

## 🧩 Project Structure
```bash
AdSentry/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependencies
├── Procfile                   # Deployment config (Heroku/HF)
├── setup.sh                   # Streamlit config for deployment
│
├── utils/
│   ├── frame_extraction.py     # Frame extraction using decord
│   ├── embeddings.py           # ViT feature extraction
│   ├── detection.py            # Cosine similarity matching
│
├── assets/
│   ├── logo.png                # Project branding
│
└── README.md
⚙️ Installation
```

----

## 📊 Output Features
🎯 Ad Match Detection

🕒 Total Advertisement Duration

📍 Timestamp Localization

📄 CSV Export

📊 Interactive Analytics UI

--- 
## 🧠 Key Features
✅ Vision Transformer (ViT) based feature extraction

✅ High-accuracy cosine similarity matching

✅ Duplicate detection filtering

✅ Customizable thresholds

✅ Streamlit-based interactive UI

✅ Real-time and offline video support

✅ Deployable on Hugging Face Spaces

---

## 💡 Use Cases
📺 Broadcast media monitoring

📊 Marketing analytics & campaign tracking

⚖️ Advertisement compliance auditing

📡 OTT platform analysis

📰 News channel ad frequency analysis

---

## 🧾 Technical Stack
Python

Streamlit

PyTorch

Hugging Face Transformers

Vision Transformer (ViT)

OpenCV

Decord

Scikit-learn

---

## 📈 Future Improvements
🔄 Real-time live stream ad detection

📊 Timeline visualization of ad occurrences

⚡ Faster inference optimization

🌐 API-based architecture (FastAPI backend)

📱 Dashboard frontend (React + Vercel)

---

## 👨‍💻 Author

Muhammad Junaid Asif (AM-Tech)  
Computer Vision and Artificial Intelligence Researcher  
📧 mjunaid94ee@outlook.com 
🌐 [[LinkedIn]](https://www.linkedin.com/in/mjunaid94ee/)  
🌐 [[Portfolio]](https://sites.google.com/view/junaid94ee/about-me)
