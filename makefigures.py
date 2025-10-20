import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# LOAD YOUR METRICS CSV
df = pd.read_csv('forgery_detection_metrics.csv')

# A. HISTOGRAM: Hamming Distance
plt.figure(figsize=(8,6))
sns.histplot(df['hamming_distance'], bins=30, kde=True, color='cornflowerblue')
plt.title('Hamming Distance Distribution')
plt.xlabel('Hamming Distance')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('hamming_distance_hist.png')
plt.close()

# B. HISTOGRAM: Cosine Similarity
plt.figure(figsize=(8,6))
sns.histplot(df['cosine_similarity'], bins=30, kde=True, color='lightsalmon')
plt.title('Cosine Similarity Distribution')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('cosine_similarity_hist.png')
plt.close()

# C. HISTOGRAM: Combined Score
plt.figure(figsize=(8,6))
sns.histplot(df['combined_score'], bins=30, kde=True, color='darkseagreen')
plt.title('Combined Score Distribution')
plt.xlabel('Combined Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('combined_score_hist.png')
plt.close()

# D. BOXPLOT: Preprocessing Time
plt.figure(figsize=(6,5))
sns.boxplot(x=df['preproc_time_db'], color='cornflowerblue')
plt.title('Preprocessing Time per Image')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig('preprocessing_time_box.png')
plt.close()

# E. BOXPLOT: Feature Extraction Time
plt.figure(figsize=(6,5))
sns.boxplot(x=df['feat_ext_time_db'], color='lightsalmon')
plt.title('Feature Extraction Time per Image')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig('feature_extraction_time_box.png')
plt.close()

# F. BOXPLOT: Hashing Time
plt.figure(figsize=(6,5))
sns.boxplot(x=df['hash_time_db'], color='darkseagreen')
plt.title('Hashing Time per Image')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig('hashing_time_box.png')
plt.close()

print('All diagrams generated and saved as PNG files in the current directory.')
