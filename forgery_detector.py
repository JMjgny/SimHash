import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_image
from hashing import compute_weighted_simhash, hamming_distance
from feature_extractor import extract_deep_features, get_feature_extractor, device
from config import HAMMING_THRESHOLD, TOP_N_MATCHES

def detect_forgery(image_path, dataset_folder, feature_extractor, 
                   hamming_threshold=HAMMING_THRESHOLD, top_n=TOP_N_MATCHES):
    """Complete forgery detection pipeline with GPU acceleration"""
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Error: Dataset folder not found {dataset_folder}")
    
    print(f"\n{'='*60}")
    print(f"Analyzing image: {os.path.basename(image_path)}")
    print(f"Using device: {device}")
    print(f"{'='*60}\n")
    
    print("Step 1: Preprocessing input image...")
    query_image_rgb, query_image_bgr = preprocess_image(image_path)
    
    print("Steps 2-5: Extracting features and computing weighted SimHash...")
    query_hash, _, _ = compute_weighted_simhash(query_image_bgr)
    query_deep_features = extract_deep_features(query_image_rgb, feature_extractor)
    
    print(f"Step 7: Comparing with dataset images...\n")
    
    matches = []
    
    for file in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file)
        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            try:
                db_image_rgb, db_image_bgr = preprocess_image(file_path)
                db_hash, _, _ = compute_weighted_simhash(db_image_bgr)
                db_deep_features = extract_deep_features(db_image_rgb, feature_extractor)
                
                ham_dist = hamming_distance(query_hash, db_hash)
                cos_sim = cosine_similarity([query_deep_features], [db_deep_features])[0][0]
                
                matches.append({
                    'filename': file,
                    'hamming_distance': ham_dist,
                    'cosine_similarity': cos_sim,
                    'combined_score': ham_dist - (cos_sim * 20)
                })
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
    
    # Clear GPU cache after processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    matches.sort(key=lambda x: x['combined_score'])
    
    print(f"\n{'='*60}")
    print(f"TOP {min(top_n, len(matches))} MATCHES:")
    print(f"{'='*60}\n")
    
    for i, match in enumerate(matches[:top_n], 1):
        print(f"Rank {i}: {match['filename']}")
        print(f"  Hamming Distance: {match['hamming_distance']}")
        print(f"  Cosine Similarity: {match['cosine_similarity']:.4f}")
        print(f"  Combined Score: {match['combined_score']:.2f}")
        
        if match['hamming_distance'] <= hamming_threshold:
            print(f"  ⚠️  POTENTIAL FORGERY DETECTED (Below threshold)")
        print()
    
    if matches:
        best_match = matches[0]
        if best_match['hamming_distance'] <= hamming_threshold:
            print(f"✓ CONCLUSION: Image is similar to '{best_match['filename']}'")
            print(f"  This suggests potential forgery or manipulation.")
        else:
            print(f"✓ CONCLUSION: No significant similarity found.")
            print(f"  Image appears to be authentic or heavily modified.")
    else:
        print("No matches found in dataset.")
    
    return matches
