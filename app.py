from flask import Flask, request, render_template
import os
from utils.preprocess import preprocess_pointcloud
from utils.inference import load_model, predict
from models.pointnetpp import PointNet2Classifier
from models.pointmlp import PointMLP
from models.dgcnn import DGCNNClassifier

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class label mapping (ModelNet10)
class_names = [
    'bathtub', 'bed', 'chair', 'desk', 'dresser',
    'monitor', 'night_stand', 'sofa', 'table', 'toilet'
]

# Load models at startup
models = {
    'pointnetpp': load_model(PointNet2Classifier, 'models/pointnetpp.pth'),
    'pointmlp': load_model(PointMLP, 'models/pointmlp.pth'),
    'dgcnn': load_model(DGCNNClassifier, 'models/DGCNN.pth')
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        model_choice = request.form['model']
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(save_path)

        pointcloud = preprocess_pointcloud(save_path)
        pred = predict(models[model_choice], pointcloud)

        # Convert class ID to class name
        class_name = class_names[pred]
        prediction = f"Predicted class: {class_name}" #(class ID: {pred})

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
