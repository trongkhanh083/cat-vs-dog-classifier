# 🐱🐶 Cat vs Dog Classifier 
https://cat-vs-dog-classifier-jw9ygsw4u5pmwhvar79ja5.streamlit.app/

A computer vision project built with TensorFlow that classifies images as **cats** or **dogs**. Trained on image datasets and deployed via a simple web app using **Streamlit**.

---

## 🚀 Features

- Custom CNN model using TensorFlow/Keras
- Data splitting and preprocessing pipeline
- Training + evaluation + prediction scripts
- Accuracy/loss curve visualization
- Web demo via Streamlit
- Easy deployment to Streamlit Cloud

## ⚙️ Installation
   ```bash
   git clone https://github.com/trongkhanh083/cat-vs-dog-classifier.git
   cd cat-vs-dog-classifier
   pip install -r requirements.txt
   ```
## 🧠 Training the Model
Run the full pipeline (data split → training → evaluation → prediction):
  ```bash
  ./scripts/run_training.sh
  ```
## 🖼️ Streamlit Web App
To run the web interface locally:
  ```bash
  streamlit run streamlit_app.py
  ```
