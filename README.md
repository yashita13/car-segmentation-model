# Car Segmentation using U-Net

## Overview
This project implements an end-to-end deep learning pipeline for car segmentation, covering dataset creation, model training, and evaluation.

Instead of relying on manually labeled data, segmentation masks were automatically generated using a YOLOv8 segmentation model. A U-Net-based architecture was then trained to perform semantic segmentation.

---

## Objective
- Build a car segmentation model using U-Net  
- Create a custom dataset using a state-of-the-art model (YOLOv8)  
- Improve performance using transfer learning and augmentation  
- Evaluate using standard segmentation metrics  

---

## Pipeline

Images → YOLOv8 (Instance Segmentation) → Mask Generation → Dataset Cleaning → U-Net Training → Evaluation

---

## Methodology

### Dataset Generation using YOLOv8
- Used YOLOv8 segmentation model (yolov8n-seg) for automatic annotation  
- Extracted polygon coordinates (`masks.xy`)  
- Converted polygons into binary masks using OpenCV (`cv2.fillPoly`)  
- Merged multiple instance masks into a single semantic mask  

### Data Preprocessing
- Resized images to 256×256  
- Normalized pixel values  
- Filtered out images with empty masks  

---

## Model Architecture

### Baseline U-Net
- Encoder: Convolution + MaxPooling layers  
- Bottleneck: High-level feature extraction  
- Decoder: Upsampling with skip connections  
- Output: Binary segmentation mask  

### Improved Model
- Encoder replaced with pretrained ResNet34  
- Enables better feature extraction through transfer learning  

---

## Loss Function

- Binary Cross Entropy (BCE): Pixel-wise classification  
- Dice Loss: Optimizes overlap between prediction and ground truth  

Combined Loss:
Loss = BCE + Dice Loss  

This helps handle class imbalance and improves segmentation performance.

---

## Models and Results

### Model 1: Baseline U-Net
- Loss: BCE  
- Epochs: 25  

Results:
- IoU: 0.586  
- Dice Score: 0.754  
- Pixel Accuracy: 0.987  

---

### Model 2: Improved U-Net
- Encoder: ResNet34 (pretrained)  
- Loss: BCE + Dice  
- Data Augmentation:
  - Horizontal Flip  
  - Rotation  
  - Brightness Adjustment  
- Epochs: 15  

Results:
- IoU: 0.803  
- Dice Score: 0.903  
- Pixel Accuracy: 0.998  

---

## Key Improvements

- Transfer learning using ResNet34 encoder  
- Data augmentation for better generalization  
- Hybrid loss function (BCE + Dice)  
- Performance comparison between models  

---

## Evaluation Metrics

- IoU (Intersection over Union): Measures overlap accuracy  
- Dice Score: Measures similarity between prediction and ground truth  
- Pixel Accuracy: Measures overall pixel correctness  

---

## Results Analysis

The improved model significantly outperformed the baseline:

- IoU improved from 0.586 to 0.803  
- Dice score improved from 0.754 to 0.903  

Key reasons:
- Better feature extraction using pretrained encoder  
- Improved generalization through augmentation  
- Use of Dice loss for overlap optimization  

---

## Visualizations

The notebook includes:
- Generated masks from YOLOv8  
- Mask overlays on input images  
- Predictions vs ground truth comparisons  
- Training loss curves  

---

## Resources

Colab Notebook:  
https://colab.research.google.com/drive/1kpSpD4VYEnz0mZec2wncHGd_C3a-PAZz?usp=sharing  

Model Weights:  
https://drive.google.com/drive/folders/1IKMC6KojIopHOFdJ4-hdXBBcUAkoUkMZ?usp=sharing  

Dataset Sample:  
https://drive.google.com/drive/folders/17OzBI3pUrNUuCFOKWzmtgXIhzBMESFgu?usp=sharing  

---

## Tech Stack

- Python  
- PyTorch  
- OpenCV  
- YOLOv8 (Ultralytics)  
- NumPy  

---

## Challenges

- Noisy pseudo-labels from YOLO-generated masks  
- Class imbalance (background dominates)  
- Some small or distant cars were missed  

Solutions:
- Used Dice loss to handle imbalance  
- Filtered out empty masks  

---

## Limitations

- Masks are pseudo-labels (not manually annotated ground truth)  
- No extensive hyperparameter tuning  
- No separate validation/test split  

---

## Future Improvements

- Hyperparameter tuning (learning rate, batch size, loss weighting)  
- Use advanced architectures (Attention U-Net, U-Net++)  
- Improve dataset quality with refined annotations  
- Add validation set and early stopping  

---

## Key Takeaway

Developed a complete segmentation pipeline and improved IoU from 0.586 to 0.803 using transfer learning, augmentation, and improved loss design.
