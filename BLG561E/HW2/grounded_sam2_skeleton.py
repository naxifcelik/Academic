import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm
import argparse

"""
Hyper parameters
"""
TEXT_PROMPT = "animal with large ears."
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
OUTPUT_DIR = Path("outputs")
DUMP_JSON_RESULTS = True
DEFAULT_VALUE = 0

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def initialize_model(device):
    # build SAM2 image predictor
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=device
    )

    return sam2_predictor, grounding_model

def process_image(img_path, sam2_predictor, grounding_model, device):
    base_name = Path(img_path).stem
    output_mask_file = OUTPUT_DIR / f"grounded_sam2_annotated_image_with_mask.jpg"
    output_json_file = OUTPUT_DIR / f"grounded_sam2_results.json"
    single_mask_file = OUTPUT_DIR / f"single_mask.png"

    # Skip processing if the mask file already exists
    if single_mask_file.exists() and output_json_file.exists():
        print(f"Skipping {img_path}, results already exist.")
        return

    # Continue processing as before
    image_source, image = load_image(img_path)
    sam2_predictor.set_image(image_source)
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device= "cpu"
    )

    # Process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes.to(device) * torch.tensor([w, h, w, h], device=device)
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.cpu().numpy().tolist()
    class_names = labels

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]
    
    print("labels:", labels)

    # Visualize image with supervision API
    img = cv2.imread(str(img_path))

    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(str(output_mask_file), annotated_frame)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(str(output_mask_file), annotated_frame)

    if DUMP_JSON_RESULTS:
        def single_mask_to_rle(mask):
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes = input_boxes.tolist()
        scores = scores.tolist()
        results = {
            "image_path": str(img_path),
            "annotations" : [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }

        with open(output_json_file, "w") as f:
            json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Grounded SAM2 Image Processing")
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID to use")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    device = "cpu"

    sam2_predictor, grounding_model = initialize_model(device)

    process_image("/home/yusuf/Desktop/ismail/animals.jpg", sam2_predictor, grounding_model, device)

if __name__ == "__main__":
    main()
