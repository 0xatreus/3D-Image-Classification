📘 README.md
# 3D Image Classification Project

A deep learning project focused on **3D object classification**, combining point cloud–based and multi-view neural network architectures.  
This repository implements **PointNet++**, **PointMLP**, and **RotationNet**, trained on the **ModelNet10/40 datasets**, and includes a **Flask web application** for serving trained models in real time.

---

## 🚀 Features
- **Multiple Architectures**  
  - *PointNet++*: Hierarchical point cloud learning  
  - *PointMLP*: Modern and efficient point cloud classifier  
  - *RotationNet*: Multi-view CNN using 12 rendered object views  

- **End-to-End Training**  
  - Implemented from scratch  
  - No pretraining or augmentation required  
  - Training logs, checkpoints, and metrics  

- **Deployment Ready**  
  - Flask app (`app.py`) with clean UI (`templates/`, `static/`)  
  - REST endpoints for model inference  
  - Easy integration into larger systems  

---

## 📂 Project Structure


3D-Image-Classification/
│── app.py # Flask web app
│── models/ # Trained models (.pth ignored by git)
│── utils/ # Helper functions (data loaders, metrics, etc.)
│── templates/ # HTML templates for Flask
│── static/ # CSS/JS assets
│── README.md # This file
│── requirements.txt # Python dependencies
│── .gitignore # Ignored files (venv, checkpoints, etc.)


---

## 🧪 Models & Accuracy
| Model        | Input Type     | Accuracy (ModelNet10) | Notes                                |
|--------------|---------------|-----------------------|--------------------------------------|
| PointNet++   | Point Cloud    | ~82–85%               | Baseline hierarchical approach        |
| PointMLP     | Point Cloud    | ~86–88%               | Most efficient & stable during tests |
| RotationNet  | Multi-View CNN | ~88–90%               | Strong accuracy, higher complexity   |

*(Exact results may vary depending on training setup and hardware.)*

---

## ⚙️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/0xatreus/3D-Image-Classification.git
   cd 3D-Image-Classification


Create a virtual environment:

python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Run the Flask app:

python app.py


App will be available at: http://127.0.0.1:5000/

📊 Dataset

ModelNet10 / ModelNet40

Popular benchmark datasets for 3D object classification.

Objects represented as meshes, point clouds, and rendered views.

Download: ModelNet Dataset

🛠️ Tech Stack

Python 3.10+

PyTorch (Deep Learning Framework)

Trimesh + Pyrender (for rendering multi-views)

Flask (Model serving)

HTML/CSS/JS (Frontend for demo)

🌟 Future Work

Add support for ModelNet40 full benchmark

Experiment with data augmentation techniques

Add Docker containerization for deployment

Integrate GitHub Actions for CI/CD

📜 License

This project is licensed under the MIT License – free to use, modify, and distribute.
