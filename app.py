import os
import torch
import kagglehub
import time
import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --- IMPROVED SYSTEM IMPORTS ---
from forgery_detector import detect_forgery_with_metrics
from feature_extractor import get_feature_extractor
from hashing import compute_weighted_simhash, hamming_distance as improved_hamming
from preprocessing import preprocess_image 

# --- PREVIOUS SYSTEM IMPORTS (Baselines) ---
# Assuming these exist in your SOP4.py as per your snippet
from Simulation import image_to_patches, md5_hash_patch, md5_to_bits, simhash_md5_frequency, hamming_distance as old_hamming

# Setup Environment
dataset_root = kagglehub.dataset_download("labid93/image-forgery-detection")
ORIGINAL_FOLDER_PATH = os.path.join(dataset_root, "DataSet/Original")
UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASET_FOLDER'] = ORIGINAL_FOLDER_PATH 
CORS(app) 

# Initialize Improved Feature Extractor (GPU if available)
feature_extractor_model = get_feature_extractor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# --- ROUTE 1: IMPROVED SYSTEM ---
@app.route('/find-original', methods=['POST'])
def handle_find_original():
    if 'tampered' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['tampered']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    
    try:
        start_time = time.time()
        # Runs the full pipeline: Preprocessing, ResNet-50, Saliency-Weighted Hashing
        matches, stats_df = detect_forgery_with_metrics(
            path, 
            ORIGINAL_FOLDER_PATH, 
            feature_extractor_model,
            top_n=1
        )
        total_time_ms = (time.time() - start_time) * 1000

        if os.path.exists(path): os.remove(path)

        if matches:
            top = matches[0]
            return jsonify({
                "system": "Improved (Thesis)",
                "status": "found",
                "filename": top['filename'],
                "preview": f"http://localhost:5000/dataset-image/{top['filename']}",
                "execution_time_ms": round(total_time_ms, 2),
                "hamming_distance": int(top['hamming_distance']),
                "cosine_similarity": round(float(top['cosine_similarity']), 4),
                "combined_score": round(float(top['combined_score']), 2)
            })
        return jsonify({"status": "not_found"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ROUTE 2: PREVIOUS SYSTEM (Baseline) ---
@app.route('/find-original-previous', methods=['POST'])
def handle_find_original_previous():
    if 'tampered' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['tampered']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    
    try:
        start_time = time.time()
        
        # 1. Load and process Query image using Previous Methods
        query_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        query_hash = simhash_md5_frequency(query_img) # Based on your first code block
        
        best_match = None
        min_dist = float('inf')
        
        # 2. Iterate through dataset (Simulating the scan_dataset_create_csv logic)
        for orig_file in os.listdir(ORIGINAL_FOLDER_PATH):
            orig_path = os.path.join(ORIGINAL_FOLDER_PATH, orig_file)
            orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            
            if orig_img is None: continue
                
            orig_hash = simhash_md5_frequency(orig_img)
            dist = old_hamming(query_hash, orig_hash)
            
            if dist < min_dist:
                min_dist = dist
                best_match = orig_file
        
        total_time_ms = (time.time() - start_time) * 1000
        
        if os.path.exists(path): os.remove(path)

        return jsonify({
            "system": "Previous (MD5-Patching)",
            "status": "found",
            "filename": best_match,
            "preview": f"http://localhost:5000/dataset-image/{best_match}",
            "execution_time_ms": round(total_time_ms, 2),
            "hamming_distance": int(min_dist),
            "cosine_similarity": None, # Previous system didn't use Deep Features
            "combined_score": int(min_dist)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/dataset-image/<filename>')
def get_dataset_image(filename):
    return send_from_directory(app.config['DATASET_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(port=5000, debug=True)