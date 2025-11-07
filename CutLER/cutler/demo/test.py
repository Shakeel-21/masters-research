import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import sys
sys.path.append('./')
sys.path.append('../')
from config import add_cutler_config
from predictor import VisualizationDemo
from PIL import Image
import imagehash
import random
# --- CONFIG AND PARSER FUNCTIONS (UNCHANGED) ---
def setup_cfg(args):
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.MODEL.DEVICE == 'cpu' and cfg.MODEL.RESNETS.NORM == 'SyncBN':
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.FPN.NORM = "BN"
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="CutLER Segmentation Mask Extractor")
    parser.add_argument("--config-file", default="model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images or a single glob pattern")
    parser.add_argument("--output", default="./output_segments", help="A directory to save output segmentation masks")
    parser.add_argument("--confidence-threshold", type=float, default=0.35, help="Minimum score for instance predictions")
    parser.add_argument("--opts", help="Modify config options", default=[], nargs=argparse.REMAINDER)
    return parser



def extract_and_save_masks_by_tile(image_path, demo, output_dir, image_hashes, crop_width=160, stride_x=160):
    """
    Processes a single image in vertical strips (full height, cropped width),
    saves unique segments, the original image,
    and visualized crops with segmentation overlays.
    """
    HAMMING_DISTANCE_THRESHOLD = 10

    img = read_image(image_path, format="BGR")
    H, W = img.shape[:2]

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(image_output_dir, exist_ok=True)

    

    new_segments_found = 0

    # --- MODIFICATION: Set y_offset and crop_y_end to full image height ---
    y_offset = 0
    crop_y_end = H

   
    for x_offset in range(0, W, stride_x):
        crop_x_end = min(x_offset + crop_width, W)
        

        current_crop = img[y_offset:crop_y_end, x_offset:crop_x_end]

        if current_crop.shape[0] == 0 or current_crop.shape[1] == 0:
            continue

        predictions, _ = demo.run_on_image(current_crop)

        if "instances" in predictions and len(predictions["instances"]) > 0:
            crop_with_overlay = current_crop.copy()
            masks_for_overlay = predictions["instances"].pred_masks.to("cpu").numpy()

            for mask in masks_for_overlay:
                color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.uint8)
                
                # --- FIX: Manually blend the color onto the masked area using NumPy ---
                crop_with_overlay[mask] = (crop_with_overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
                # --------------------------------------------------------------------

            # --- Filename still includes y_offset (which is 0) for consistency ---
            

        if "instances" not in predictions: continue
        instances = predictions["instances"].to("cpu")
        if len(instances) == 0: continue

        masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy().astype(int)

        for idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            x1_rel, y1_rel, x2_rel, y2_rel = box
            if x2_rel <= x1_rel or y2_rel <= y1_rel: continue

            segment_part = current_crop[y1_rel:y2_rel, x1_rel:x2_rel].copy()
            cropped_mask = mask[y1_rel:y2_rel, x1_rel:x2_rel]

            masked_segment = segment_part.copy()
            masked_segment[~cropped_mask.astype(bool)] = 0

            try:
                pil_image = Image.fromarray(cv2.cvtColor(masked_segment, cv2.COLOR_BGR2RGB))
                current_hash = imagehash.dhash(pil_image)

                is_duplicate = False
                for existing_hash in image_hashes:
                    if (current_hash - existing_hash) < HAMMING_DISTANCE_THRESHOLD:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    image_hashes.append(current_hash)

                    segment_filename = f"{base_name}_seg_{new_segments_found}.png"
                    segment_path = os.path.join(image_output_dir, segment_filename)
                    cv2.imwrite(segment_path, masked_segment)
                    new_segments_found += 1

            except Exception as e:
                print(f"Error processing a segment for hashing/saving: {e}")
                continue

    print(f"{base_name}: saved {new_segments_found} new unique cropped segments.")
    return len(image_hashes)

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    os.makedirs(args.output, exist_ok=True)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        logger.info(f"Processing {len(args.input)} images...")
        logger.info(f"Output directory: {args.output}")
        
        for path in tqdm.tqdm(args.input):
            start_time = time.time()
            
            # --- KEY CHANGE: Create a new, empty hash list for each image ---
            hashes_for_this_image = []
            
            # --- Pass this specific list to the function ---
            num_unique_instances = extract_and_save_masks_by_tile(
                path, demo, args.output, hashes_for_this_image
            )

            logger.info(
                "{}: found {} total unique instances in {:.2f}s".format(
                    path,
                    num_unique_instances,
                    time.time() - start_time,
                )
            )

        logger.info(f"\nDone! All segmentation masks saved to: {args.output}/")
    else:
        logger.error("Please specify --input")