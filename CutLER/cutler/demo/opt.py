import argparse
import pathlib
import numpy as np
import imagehash
import cv2
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm

def get_image_features(image_paths, method='phash', min_width=13):
    """
    Calculates the feature (hash or histogram) for each image,
    filtering out images smaller than min_width.
    """
    features = []
    valid_paths = []
    
    print(f"Calculating features for all images using method: {method}...")
    
    for path in tqdm(image_paths, desc="Extracting Features"):
        try:
            if method in ['phash', 'dhash']:
                with Image.open(path) as img:
                    # --- NEW FILTER ---
                    width, _ = img.size
                    if width < min_width:
                        continue
                    # --- END FILTER ---
                        
                    if method == 'phash':
                        feat = imagehash.phash(img)
                    elif method == 'dhash':
                        feat = imagehash.dhash(img)
                features.append(feat)
                valid_paths.append(path)
            
            elif method == 'histogram':
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read {path} with OpenCV. Skipping.")
                    continue
                
                # --- NEW FILTER ---
                height, width = img.shape[:2]
                if width < min_width:
                    continue
                # --- END FILTER ---
                
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                
                features.append(hist)
                valid_paths.append(path)
                
        except Exception as e:
            print(f"Warning: Could not process {path}: {e}. Skipping.")
            
    return features, valid_paths

def calculate_distance_matrix(features, method='phash'):
    """
    Calculates the N x N distance matrix based on the chosen method.
    """
    n_images = len(features)
    print("Calculating distance matrix...")
    distance_matrix = np.zeros((n_images, n_images), dtype=float)

    for i in tqdm(range(n_images), desc="Calculating Distances"):
        for j in range(i, n_images):
            if i == j:
                dist = 0.0
            else:
                if method in ['phash', 'dhash']:
                    dist = float(features[i] - features[j])
                elif method == 'histogram':
                    dist = cv2.compareHist(features[i], features[j], cv2.HISTCMP_BHATTACHARYYA)
            
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
            
    return distance_matrix

def find_best_k(input_dir, method, dedup=False, min_clusters=2, max_clusters=15, min_width=13):
    """
    Finds features and calculates silhouette scores for a range of k values.
    """
    input_path = pathlib.Path(input_dir)

    # 1. Find all PNG images
    print("Finding all .png images...")
    image_paths = list(input_path.rglob('*.png'))
    if not image_paths:
        print("No .png images found.")
        return

    print(f"Found {len(image_paths)} images.")

    # 2. Get features, filtering by width
    features, valid_paths = get_image_features(image_paths, method, min_width)
    print(f"Filtered to {len(valid_paths)} images (width >= {min_width}).")

    # 3. Deduplication step
    if dedup:
        if method in ['phash', 'dhash']:
            print("Filtering for duplicate images using hashes...")
            unique_features = []
            unique_paths = []
            seen_hashes = set()
            
            for feat, path in zip(features, valid_paths):
                if feat not in seen_hashes:
                    seen_hashes.add(feat)
                    unique_features.append(feat)
                    unique_paths.append(path)
            
            print(f"Filtered count (unique images): {len(unique_paths)}")
            features = unique_features
            valid_paths = unique_paths
        else:
            print(f"Warning: Deduplication is only supported for 'phash' or 'dhash'. Skipping for '{method}'.")

    n_images = len(valid_paths)

    if n_images < max_clusters:
        print(f"Warning: Number of images ({n_images}) is less than max_clusters ({max_clusters}).")
        max_clusters = n_images - 1

    if n_images < min_clusters:
        print(f"Error: Not enough images ({n_images}) to test min_clusters ({min_clusters}).")
        return

    # 4. Calculate distance matrix
    distance_matrix = calculate_distance_matrix(features, method)

    # 5. Loop through k values
    print(f"\n--- Calculating Silhouette Scores (Method: {method}) ---")
    best_score = -1
    best_k = 0

    k_values = range(min_clusters, max_clusters + 1)
    
    for k in tqdm(k_values, desc="Testing k values"):
        cluster_model = AgglomerativeClustering(
            n_clusters=k, 
            affinity='precomputed', 
            linkage='average'
        )
        labels = cluster_model.fit_predict(distance_matrix)
        
        try:
            score = silhouette_score(distance_matrix, labels, metric='precomputed')
            print(f"Clusters (k) = {k}: Silhouette Score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        except ValueError as e:
            print(f"Clusters (k) = {k}: Could not calculate score (likely only 1 cluster found). Error: {e}")

    print("---------------------------------")
    print(f"\nDone! Best number of clusters (k) is {best_k} with a score of {best_score:.4f}")
    print(f"You can now run your main script with the argument: -n {best_k}")


def main():
    parser = argparse.ArgumentParser(description="Find the optimal number of clusters (k) using the Silhouette Score.")
    
    parser.add_argument(
        "input_folder", 
        type=str,
        help="The source folder containing .png images (can be nested)."
    )
    
    parser.add_argument(
        "--min",
        type=int,
        default=2,
        help="The minimum number of clusters to test."
    )
    
    parser.add_argument(
        "--max",
        type=int,
        default=15,
        help="The maximum number of clusters to test."
    )

    parser.add_argument(
        "-m", "--method",
        type=str,
        default="phash",
        choices=['phash', 'dhash', 'histogram'],
        help="The similarity method to use (default: 'phash')."
    )

    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Enable filtering of duplicate images (only works with 'phash' or 'dhash')."
    )
    
    # --- ADDED THIS ARGUMENT ---
    parser.add_argument(
        "--min_width",
        type=int,
        default=13,
        help="Filter out images with width smaller than this value (default: 13)."
    )
    # --- END OF ADDITION ---
    
    args = parser.parse_args()

    if args.min < 2:
        print("Error: Minimum clusters must be 2 or more.")
        return

    find_best_k(
        args.input_folder, 
        args.method, 
        args.dedup, 
        args.min, 
        args.max, 
        args.min_width
    )

if __name__ == "__main__":
    main()