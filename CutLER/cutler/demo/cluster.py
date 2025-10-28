import argparse
import pathlib
import shutil
import numpy as np
import imagehash
import cv2
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
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
                # OpenCV shape is (height, width)
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

def run_clustering(input_dir, output_dir, n_clusters, method, dedup=False, min_width=13):
    """
    Finds, features, and clusters all PNG images in a directory.
    """
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)

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
    if n_images < n_clusters:
        print(f"Error: Found only {n_images} unique images, < {n_clusters} clusters.")
        return

    # 4. Calculate pairwise distance
    distance_matrix = calculate_distance_matrix(features, method)

    # 5. Perform clustering
    print("Clustering images...")
    cluster_model = AgglomerativeClustering(
        n_clusters=n_clusters, 
        affinity='precomputed', # Using 'affinity' for your sklearn version
        linkage='average'
    )
    
    labels = cluster_model.fit_predict(distance_matrix)

    # 6. Save *original* images into new cluster folders
    print(f"Saving images into {n_clusters} cluster folders...")
    if output_path.exists():
        shutil.rmtree(output_path)
    
    for i in range(n_clusters):
        cluster_dir = output_path / f"cluster_{i}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

    for idx, (original_path, label) in enumerate(zip(valid_paths, labels)):
        target_dir = output_path / f"cluster_{label}"
        new_name = f"{idx}_{original_path.name}"
        shutil.copy(original_path, target_dir / new_name)

    print(f"\nDone! Clustered {n_images} images into {n_clusters} bins at '{output_path}'.")

def main():
    parser = argparse.ArgumentParser(description="Cluster PNG images based on visual similarity.")
    
    parser.add_argument(
        "input_folder", 
        type=str,
        help="The source folder containing .png images (can be nested)."
    )
    
    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        default="image_clusters",
        help="The directory to save the clustered image folders (default: 'image_clusters')."
    )
    
    parser.add_argument(
        "-n", "--bins",
        type=int,
        default=5,
        help="The number of clusters (bins) to create (default: 5)."
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

    if args.bins <= 0:
        print("Error: Number of bins must be greater than 0.")
        return

    run_clustering(
        args.input_folder, 
        args.output_folder, 
        args.bins, 
        args.method, 
        args.dedup, 
        args.min_width
    )

if __name__ == "__main__":
    main()