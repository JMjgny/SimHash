import os
import torch
import kagglehub
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from forgery_detector import detect_forgery_with_metrics
from feature_extractor import get_feature_extractor


print("="*60)
print("SERVER STARTING UP: Performing one-time setup...")
print("="*60)


print("GPU CONFIGURATION")
print("="*60)
if torch.cuda.is_available():
    print(f"✓ CUDA is available")
    print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("✗ CUDA is NOT available. Using CPU.")
print("="*60 + "\n")

print("Finding/Downloading Kaggle dataset...")
dataset_root = kagglehub.dataset_download("labid93/image-forgery-detection")
ORIGINAL_FOLDER_PATH = os.path.join(dataset_root, "DataSet/Original")
print(f"✓ Dataset ready. Originals at: {ORIGINAL_FOLDER_PATH}")


print("Loading feature extraction model (this may take a moment)...")
feature_extractor_model = get_feature_extractor()
print("✓ Model loaded and ready.")
print("="*60)
print("SERVER IS READY TO RECEIVE REQUESTS!")
print("="*60)



UPLOAD_FOLDER = 'uploads' 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASET_FOLDER'] = ORIGINAL_FOLDER_PATH 
CORS(app) 


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def find_best_match(uploaded_image_path):
    """
    This helper function runs your detection logic.
    """
    try:
        matches, stats = detect_forgery_with_metrics(
            uploaded_image_path,
            ORIGINAL_FOLDER_PATH,
            feature_extractor_model
        )
       
        if not matches:
            print("No matches returned from detection logic.")
            return None 


        top_match = matches[0] 
        match_score = top_match.get('combined_score')
        match_filename = top_match.get('filename')


        if match_score is None or match_filename is None:
            print(f"Error: Top match dict is missing keys. Got: {top_match}")
            return None
        THRESHOLD = 4.0
       
        print(f"Top match found: {match_filename} with Score: {match_score}")
       
        if match_score < THRESHOLD:
            print("✓ Score is below threshold. This is a match.")
            return match_filename
        else:
            print(f"✗ Match score ({match_score}) is not below threshold ({THRESHOLD}).")
            return None
           
    except Exception as e:
        print(f"Error during detect_forgery_with_metrics: {e}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ GPU memory cache cleared")
@app.route('/find-original', methods=['POST'])
def handle_find_original():
    """
    This is "Door #1".
    The HTML's JavaScript calls this when you click the "Find" button.
    """
   
    if 'tampered' not in request.files:
        return jsonify({"error": "No file part"}), 400
   
    file = request.files['tampered'] 
   
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_image_path)
        match_filename = find_best_match(uploaded_image_path)
        if os.path.exists(uploaded_image_path):
            os.remove(uploaded_image_path)
        if match_filename:
            preview_url = f"http://localhost:5000/dataset-image/{match_filename}"
            response_data = {
                "status": "found",
                "filename": match_filename,
                "preview": preview_url 
            }
            return jsonify(response_data)
        else:
            response_data = { "status": "not_found" }
            return jsonify(response_data)
               
    return jsonify({"error": "File type not allowed"}), 400


@app.route('/dataset-image/<filename>')
def get_dataset_image(filename):
    """
    This is "Door #2".
    The HTML page calls this to get the preview image of the match.
    """
    return send_from_directory(app.config['DATASET_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(port=5000, debug=True)