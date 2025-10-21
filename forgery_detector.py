import os
import torch
import numpy as np
import time
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_image
from hashing import compute_weighted_simhash, hamming_distance
from feature_extractor import extract_deep_features,get_feature_extractor, device

def detect_forgery_with_metrics(image_path, dataset_folder, feature_extractor, 
                               hamming_threshold=15, top_n=1):
    """
    Enhanced forgery detection pipeline with detailed timing and similarity logging.
    Returns:
      matches: List of top N matches with similarity scores
      df_stats: Pandas DataFrame of detailed per-image metrics for statistical analysis
    """
    print(f"\n{'='*60}")
    print(f"Analyzing image: {os.path.basename(image_path)}")
    print(f"Using device: {device}")
    print(f"{'='*60}\n")

    stats = defaultdict(list)
    error_count = 0
    matches = []

    # Preprocess query image
    start = time.time()
    query_rgb, query_bgr = preprocess_image(image_path)
    stats['preproc_time_query'].append(time.time() - start)

    # Extract features query image
    start = time.time()
    query_features = extract_deep_features(query_rgb, feature_extractor)
    stats['feat_ext_time_query'].append(time.time() - start)

    # Compute hash query image
    start = time.time()
    query_hash, _, _ = compute_weighted_simhash(query_bgr)
    stats['hash_time_query'].append(time.time() - start)

    # List all images in dataset
    image_files = [f for f in os.listdir(dataset_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"Comparing against {len(image_files)} images in dataset...")

    # Timings for dataset processing
    preproc_times_db = []
    feat_ext_times_db = []
    hash_times_db = []

    for filename in image_files:
        try:
            file_path = os.path.join(dataset_folder, filename)

            # Preprocessing time
            start = time.time()
            db_rgb, db_bgr = preprocess_image(file_path)
            preproc_time = time.time() - start
            preproc_times_db.append(preproc_time)

            # Feature extraction time
            start = time.time()
            db_features = extract_deep_features(db_rgb, feature_extractor)
            feat_ext_time = time.time() - start
            feat_ext_times_db.append(feat_ext_time)

            # Hashing time
            start = time.time()
            db_hash, _, _ = compute_weighted_simhash(db_bgr)
            hash_time = time.time() - start
            hash_times_db.append(hash_time)

            # Compute similarity metrics
            ham_dist = hamming_distance(query_hash, db_hash)
            cos_sim = cosine_similarity([query_features], [db_features])[0][0]
            combined_score = ham_dist - (cos_sim * 20)

            matches.append({
                'filename': filename,
                'hamming_distance': ham_dist,
                'cosine_similarity': cos_sim,
                'combined_score': combined_score
            })

            # Log to stats for this comparison
            stats['filename'].append(filename)
            stats['hamming_distance'].append(ham_dist)
            stats['cosine_similarity'].append(cos_sim)
            stats['combined_score'].append(combined_score)
            stats['preproc_time_db'].append(preproc_time)
            stats['feat_ext_time_db'].append(feat_ext_time)
            stats['hash_time_db'].append(hash_time)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            error_count += 1
            # Append None or NaN to keep list lengths equal
            stats['filename'].append(filename)
            stats['hamming_distance'].append(None)
            stats['cosine_similarity'].append(None)
            stats['combined_score'].append(None)
            stats['preproc_time_db'].append(None)
            stats['feat_ext_time_db'].append(None)
            stats['hash_time_db'].append(None)
            continue

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
            print(f"  ⚠️ POTENTIAL FORGERY DETECTED (Below threshold)")
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Debug: Check that all lists are the same length before DataFrame creation
    print("Lengths of stats lists:")
    for key, lst in stats.items():
        print(f"{key}: {len(lst)}")

    stats_for_df = {k: v for k, v in stats.items() if not k.endswith('_query')}
    # Save per-image metrics only (exclude averaged lists)
    df_stats = pd.DataFrame(stats_for_df)

    csv_path = 'forgery_detection_metrics.csv'
    df_stats.to_csv(csv_path, index=False)
    print(f"\nSaved detailed metrics to {csv_path}")

    # Optionally, print query-level preprocessing and feature extraction times separately
    print("Query-level times:")
    for key in ['preproc_time_query', 'feat_ext_time_query', 'hash_time_query']:
        print(f"{key}: {stats[key][0]:.4f} seconds")

    return matches, df_stats

def run_improved_pairwise(original_img_path, forged_img_path):
    # Initialize feature extractor once (you may want to cache this outside if comparing many pairs)
    feature_extractor = get_feature_extractor()

    # Preprocess images
    orig_rgb, orig_bgr = preprocess_image(original_img_path)
    forged_rgb, forged_bgr = preprocess_image(forged_img_path)

    # Extract deep features
    orig_features = extract_deep_features(orig_rgb, feature_extractor)
    forged_features = extract_deep_features(forged_rgb, feature_extractor)

    # Compute weighted simhash
    orig_hash, _, _ = compute_weighted_simhash(orig_bgr)
    forged_hash, _, _ = compute_weighted_simhash(forged_bgr)

    # Compute distances
    hamm_dist = hamming_distance(orig_hash, forged_hash)
    cos_sim = np.dot(orig_features, forged_features) / (np.linalg.norm(orig_features) * np.linalg.norm(forged_features))

    # Prepare a concise result summary string
    result_summary = (
        f"Improved Algorithm Results:\n"
        f"Hamming Distance: {hamm_dist}\n"
        f"Cosine Similarity: {cos_sim:.4f}\n"
    )

    return result_summary
