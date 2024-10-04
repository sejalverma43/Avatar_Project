import argparse
import torch
from PIL import Image
import numpy as np
import cv2
import os
import sys

# Add GroundingDINO and Segment Anything to the Python path
sys.path.append(os.path.join(os.getcwd(), 'GroundingDINO'))
sys.path.append(os.path.join(os.getcwd(), 'segment-anything'))

from groundingdino.util.inference import load_model, predict
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInpaintPipeline
from torchvision.ops import box_convert

def load_groundingdino_model(device):
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    ckpt_file = "models/groundingdino_swinb_cogcoor.pth"
    model = load_model(config_file, ckpt_file, device=device)
    return model

def load_sam_model(device):
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--class', type=str, dest='cls', required=True, help='Object class to segment.')
    parser.add_argument('--output', type=str, default='output.png', help='Path to save the output image.')
    parser.add_argument('--x', type=int, default=0, help='Pixel shift in x-direction.')
    parser.add_argument('--y', type=int, default=0, help='Pixel shift in y-direction.')
    parser.add_argument('--box_threshold', type=float, default=0.3, help='Box threshold for object detection.')
    parser.add_argument('--text_threshold', type=float, default=0.25, help='Text threshold for object detection.')
    args = parser.parse_args()

    # Determine the device to run on
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load image
    image_pil = Image.open(args.image).convert("RGB")
    image_np = np.array(image_pil)

    # Load models
    dino_model = load_groundingdino_model(device)
    sam_predictor = load_sam_model(device)

    # Predict bounding boxes with GroundingDINO
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_pil,
        caption=args.cls,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device
    )

    if len(boxes) == 0:
        print(f"No objects found for class '{args.cls}'.")
        return

    # Prepare image for SAM
    sam_predictor.set_image(image_np)

    # Transform boxes for SAM
    boxes_xyxy = boxes.to(device) * torch.tensor(
        [image_np.shape[1], image_np.shape[0], image_np.shape[1], image_np.shape[0]], 
        dtype=torch.float32, 
        device=device
    )
    boxes_xyxy = box_convert(boxes_xyxy, in_fmt='cxcywh', out_fmt='xyxy')
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_np.shape[:2])

    # Generate masks with SAM
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    mask = masks[0][0].to(device).bool()

    if args.x == 0 and args.y == 0:
        red_mask = torch.zeros_like(torch.from_numpy(image_np).to(device), dtype=torch.uint8)
        red_mask[mask] = torch.tensor([255, 0, 0], device=device, dtype=torch.uint8)

        # Combine the original image with the red mask overlay
        output_image = torch.clamp(torch.from_numpy(image_np).to(device).float() * 1 + red_mask.float() * 0.5, 0, 255).byte().cpu().numpy()
        output_pil = Image.fromarray(output_image)
        output_pil.save(args.output)
        print(f"Output saved to {args.output}")
    else:
        if device == 'cuda':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch_dtype,
        ).to(device)

        init_image = Image.fromarray(image_np).resize((512, 512))
        init_image_np = np.array(init_image)

        mask_resized = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8)).resize((512, 512))
        mask_resized_np = np.array(mask_resized) / 255.0

        mask_image = Image.fromarray((mask_resized_np * 255).astype(np.uint8))

        prompt = "Background"
        with torch.amp.autocast(device_type=device):
            inpaint_result = inpaint_pipe(
                prompt=prompt,
                image=init_image,
                mask_image=mask_image,
            ).images[0]

        # Composite the images
        inpaint_result_np = np.array(inpaint_result.resize((image_np.shape[1], image_np.shape[0])))
        final_image = inpaint_result_np.copy()

        # Shift the object using OpenCV warpAffine
        object_image = image_np * mask.cpu().numpy()[:, :, None]  # Object image based on mask
        M = np.float32([[1, 0, args.x], [0, 1, -args.y]])  # Shifting matrix
        shifted_object = cv2.warpAffine(
            object_image, M, (object_image.shape[1], object_image.shape[0]),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        shifted_mask = cv2.warpAffine(
            mask.cpu().numpy().astype(np.uint8), M, (mask.shape[1], mask.shape[0]),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        ).astype(bool)

        # Ensure the object and mask are in the correct shape
        shifted_object = shifted_object  # Object has already been shifted
        shifted_mask = shifted_mask  # Mask has already been shifted

        # Resize shifted object and mask if their dimensions do not match the final image
        if shifted_object.shape[:2] != final_image.shape[:2]:
            shifted_object = cv2.resize(shifted_object, (final_image.shape[1], final_image.shape[0]))

        if shifted_mask.shape != final_image.shape[:2]:
            shifted_mask = cv2.resize(shifted_mask.astype(np.uint8), (final_image.shape[1], final_image.shape[0])).astype(bool)

        # Apply shifted object to final image using the mask
        final_image[shifted_mask] = shifted_object[shifted_mask]

        # Save the final image
        output_pil = Image.fromarray(final_image)
        output_pil.save(args.output)
        print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()