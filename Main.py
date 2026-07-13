import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from IPython.display import Video
from google.colab import files
from ultralytics import YOLO
import time  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384
NUM_CLASSES = 19
ROAD_CLASS = 0


SMP_MODEL_PATH = "/content/unetpp_effb3.pth"
YOLO_MODEL_PATH = "/content/best.pt"
VIDEO_PATH = "/content/input_video.mp4"
OUTPUT_PATH = "/content/day_output.mp4"

# DEFINING THE COLOR PALETTE OF PIXELS FOR BETTER VISUALIZATION
PALETTE = np.array([
    [0, 100, 0],      # Class 0: Road/Lanes (Dark Green)
    [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

# DEFINING THE SEGMENTATION MODEL
smp_model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b3",
    encoder_weights=None,
    in_channels=3,
    classes=19,
    activation=None
)

ckpt = torch.load(SMP_MODEL_PATH, map_location=device)

# Robust fallback dictionary checks for different checkpoint formats
if isinstance(ckpt, dict):
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
else:
    state_dict = ckpt

# LOADING THE TRAINED SEGMENTATION MODEL
smp_model.load_state_dict(state_dict)
smp_model.to(device)
smp_model.eval()

# LOADING THE YOLO MODEL
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.to(device)

# Same Transforms for Segmentation which were used while model training
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Video Setup 
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Inference & Tracking Loop 
for _ in tqdm(range(frames)):
    ret, frame = cap.read()
    if not ret:
        break

    # Start timer for FPS calculation
    start_time = time.time()

    # Running YOLO Object Tracking via ByteTrack
    yolo_results = yolo_model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Running Unet++ Road Segmentation 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    aug = transform(image=rgb)
    x = aug["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = smp_model(x)
        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

    # Resize mask to original input dimensions
    pred = cv2.resize(
        pred.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    # Building Dark Green Mask on the Road/Lane
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    color_mask[pred == ROAD_CLASS] = PALETTE[ROAD_CLASS]

    # Overlay mask on background frame
    overlay = cv2.addWeighted(
        frame,
        0.50,
        color_mask,
        0.50,
        0
    )

    # JUST FOR BEAUTIFUL VISUALIZATION AND LESS LABEL'S CLUTTER
    if yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
        final_output = yolo_results[0].plot(img=overlay, labels=False, conf=True, boxes=True)
    else:
        final_output = overlay

    # Calculating and Drawing FPS 
    end_time = time.time()
    processing_fps = 1.0 / (end_time - start_time)

    # Text configuration
    fps_text = f"FPS: {processing_fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    text_color = (255, 255, 255)
    thickness = 2

    # Rendering thin black shadow outline behind the text for visibility
    cv2.putText(final_output, fps_text, (20, 50), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(final_output, fps_text, (20, 50), font, font_scale, text_color, thickness, cv2.LINE_AA)

    writer.write(final_output)

cap.release()
writer.release()

print("Finished!")

# Display Output
Video(OUTPUT_PATH, embed=True)