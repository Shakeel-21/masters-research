import argparse
import pathlib
import shutil
import sys
import cv2
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import multiprocessing

# --- CONFIGURATION ---
# Thresholds - Tweak these if you get too many/too few results
HIST_SIMILARITY_THRESHOLD = 0.1  # Lower = stricter match (0.0 is identical)
TEMPLATE_MATCH_THRESHOLD = 0.15   # For SQDIFF: Lower is better (0.0 is perfect match)

class ImageUtils:
    """Static utility class for file and image handling."""
    
    @staticmethod
    def load_bgra(path: pathlib.Path):
        """Loads an image ensuring 4 channels (BGRA)."""
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None: return None
        
        # Convert to BGRA if it happens to be grayscale or BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

    @staticmethod
    def load_bgr(path: pathlib.Path):
        """Loads an image ensuring 3 channels (BGR)."""
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

    @staticmethod
    def calc_masked_histogram(img_bgra):
        """
        Calculates a color histogram IGNORING transparent pixels.
        This is crucial for accurate sprite deduplication.
        """
        # Split channels
        b, g, r, a = cv2.split(img_bgra)
        
        # Calculate hist for B, G, R using Alpha as mask
        hist_features = []
        for channel in [b, g, r]:
            # Calc hist: [image], [channel], mask, [bins], [range]
            hist = cv2.calcHist([channel], [0], a, [256], [0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hist_features.extend(hist.flatten())
            
        return np.array(hist_features, dtype=np.float32)

class SpriteProcessor:
    """
    Manages the ingestion, cleaning (deduplication), and clustering 
    of the small segment images.
    """
    def __init__(self, input_dir, min_width=13):
        self.input_dir = pathlib.Path(input_dir)
        self.min_width = min_width
        self.raw_paths = []
        self.unique_data = [] # List of tuples: (path, image_data, histogram)

    def load_and_process_histograms(self):
        """
        Loads all images and computes histograms in parallel.
        """
        print(f"Scanning {self.input_dir}...")
        self.raw_paths = list(self.input_dir.rglob("*.png"))
        
        if not self.raw_paths:
            print("No PNG images found.")
            return

        print(f"Computing histograms for {len(self.raw_paths)} images (Parallel)...")
        
        # Helper for parallel execution
        def _process_single(path):
            img = ImageUtils.load_bgra(path)
            if img is None: return None
            if img.shape[1] < self.min_width: return None # Width filter
            
            hist = ImageUtils.calc_masked_histogram(img)
            return (path, img, hist)

        # Execute on all cores
        n_jobs = multiprocessing.cpu_count()
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_single)(p) for p in tqdm(self.raw_paths, desc="Analyzing")
        )
        
        # Filter failures
        self.clean_results = [r for r in results if r is not None]

    def deduplicate(self):
        """
        Filters the loaded images using the Histogram comparison method.
        """
        if not hasattr(self, 'clean_results'):
            self.load_and_process_histograms()

        unique_items = [] # Stores (path, img, hist)
        
        print(f"Deduplicating {len(self.clean_results)} images...")
        
        # Note: We cannot easily parallelize the comparison because it depends on 
        # the growing list of 'unique_items'.
        for new_item in tqdm(self.clean_results, desc="Filtering Duplicates"):
            path, img, hist = new_item
            is_duplicate = False
            
            for _, _, u_hist in unique_items:
                # Correlation: 1.0 is perfect match. Bhattacharyya: 0.0 is perfect match.
                # We use Bhattacharyya for accuracy.
                # We reshape to float32 for OpenCV compatibility if needed, though numpy handles it.
                score = cv2.compareHist(hist, u_hist, cv2.HISTCMP_BHATTACHARYYA)
                
                if score < HIST_SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_items.append(new_item)

        self.unique_data = unique_items
        print(f" Reduction: {len(self.clean_results)} -> {len(self.unique_data)} unique sprites.")

    def cluster_and_save(self, output_dir, n_clusters=5):
        """
        Clusters the UNIQUE images and saves them to disk organized by similarity.
        Returns the path to the organized folder for reference.
        """
        if len(self.unique_data) < n_clusters:
            print("Not enough unique images to cluster. Skipping clustering step.")
            return

        print("Clustering unique images for organization...")
        
        # 1. Build Distance Matrix (N x N)
        feats = [x[2] for x in self.unique_data]
        n = len(feats)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = cv2.compareHist(feats[i], feats[j], cv2.HISTCMP_BHATTACHARYYA)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # 2. Cluster
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
        labels = model.fit_predict(dist_matrix)

        # 3. Save
        out_path = pathlib.Path(output_dir)
        if out_path.exists(): shutil.rmtree(out_path)
        
        for idx, label in enumerate(labels):
            original_path, img, _ = self.unique_data[idx]
            target_dir = out_path / f"cluster_{label}"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Save matching the original name
            cv2.imwrite(str(target_dir / original_path.name), img)
            
        print(f"Clustered sprites saved to: {out_path}")

class SceneReconstructor:
    """
    Takes a list of unique sprites and matches them into source level images.
    """
    def __init__(self, level_dir, output_dir):
        self.level_dir = pathlib.Path(level_dir)
        self.output_dir = pathlib.Path(output_dir)

    def match_sprites(self, unique_sprites):
        """
        Args:
            unique_sprites: List of tuples (path, image_bgra, histogram)
        """
        level_paths = list(self.level_dir.glob("*.png"))
        print(f"\nMatching {len(unique_sprites)} unique sprites into {len(level_paths)} levels...")

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        # Parallelize by Level (Process each level on a separate core)
        n_jobs = multiprocessing.cpu_count()
        
        Parallel(n_jobs=n_jobs)(
            delayed(self._process_level)(l_path, unique_sprites) 
            for l_path in tqdm(level_paths, desc="Reconstructing")
        )
        
        print(f"\nReconstruction complete. Results in: {self.output_dir}")

    def _process_level(self, level_path, unique_sprites):
        """
        Internal worker function to process a single level.
        """
        level_img = ImageUtils.load_bgr(level_path)
        if level_img is None: return

        level_name = level_path.stem
        level_out = self.output_dir / level_name
        matches_found = 0

        for (t_path, t_img_bgra, _) in unique_sprites:
            
            # 1. Dimensionality Check (Critical Optimization)
            h, w = t_img_bgra.shape[:2]
            lh, lw = level_img.shape[:2]
            if h > lh or w > lw: continue

            # 2. Extract Mask
            mask = t_img_bgra[:, :, 3]
            if cv2.countNonZero(mask) == 0: continue # Skip empty sprites

            # 3. Convert Template to BGR for matching
            t_bgr = cv2.cvtColor(t_img_bgra, cv2.COLOR_BGRA2BGR)

            try:
                # 4. Match Template (Squared Difference is best for exact sprite matches)
                res = cv2.matchTemplate(level_img, t_bgr, cv2.TM_SQDIFF, mask=mask)
                min_val, _, _, _ = cv2.minMaxLoc(res)

                # 5. Normalize score roughly based on sprite size to make threshold consistent
                # (Optional, but helps if sprites vary wildly in size)
                # Here we stick to raw SQDIFF. 
                # If min_val is very close to 0, it's a match.
                
                # However, SQDIFF can be large numbers. Let's use Normed SQDIFF if needed,
                # but for speed, we usually stick to simple thresholding.
                # Let's assume the user wants "Visual Match".
                
                # Simple heuristic: For SQDIFF, 0 is perfect.
                # We can roughly normalize by area to make the threshold generic.
                norm_score = min_val / (w * h)
                
                if norm_score < TEMPLATE_MATCH_THRESHOLD:
                    matches_found += 1
                    level_out.mkdir(parents=True, exist_ok=True)
                    # Save the original BGRA sprite back to the reconstruction folder
                    cv2.imwrite(str(level_out / t_path.name), t_img_bgra)

            except Exception as e:
                continue
        
        # Only log if we found something (avoids messy tqdm output)
        if matches_found > 0:
            pass 

def main():
    parser = argparse.ArgumentParser(description="Unified Sprite Clustering & Reconstruction Pipeline")
    
    parser.add_argument("sprites_dir", help="Input folder containing raw sprite segments")
    parser.add_argument("levels_dir", help="Input folder containing source level images")
    
    parser.add_argument("--cluster_out", default="output_clusters", help="Folder to save organized clusters")
    parser.add_argument("--recon_out", default="output_reconstructed", help="Folder to save matched results")
    
    parser.add_argument("--bins", type=int, default=7, help="Number of clusters for organization")
    parser.add_argument("--min_width", type=int, default=13, help="Minimum sprite width to process")

    args = parser.parse_args()

    # 1. Initialize Processor
    processor = SpriteProcessor(args.sprites_dir, min_width=args.min_width)

    # 2. Load & Deduplicate (The heavy lifting)
    # This fills processor.unique_data with the clean set
    processor.deduplicate()

    # 3. Cluster (Organize the clean set)
    processor.cluster_and_save(args.cluster_out, n_clusters=args.bins)

    # 4. Match (Reconstruct using the clean set)
    matcher = SceneReconstructor(args.levels_dir, args.recon_out)
    # We pass the unique data from the processor to the matcher
    matcher.match_sprites(processor.unique_data)

if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support() 
    main()