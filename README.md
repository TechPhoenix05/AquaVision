# MicroPi – An AI-powered Microscopy for Real-time Marine Organism Analysis
**Team Name:** Tech Phoenix  
**Team ID:** 67713  
**Theme:** Smart Automation  
**PS Category:** Hardware  
**College:** A.V.C College of Engineering, Mayiladuthurai, Tamil Nadu  

---

## Team Members
1. P.M. Thirumaran (Team Leader)  
2. V. Sahana  
3. B. Azrina  
4. S. Praveen  
5. R. Dhaneshwar  
6. B. Vigneshkumaran  

---

## Abstract
The detection and monitoring of phytoplanktons and zooplanktons are crucial for assessing marine ecosystem health, water quality, and identifying harmful algal blooms. Traditional laboratory-based identification is slow, costly, and unsuitable for continuous field monitoring.

MicroPi proposes a low-cost, real-time plankton detection system using Raspberry Pi 4 with a high-resolution camera and image-processing algorithms. Raspberry Pi captures underwater microscopic images and processes them locally using OpenCV and machine learning to classify plankton based on shape, size, and texture.

The system provides onboard storage, wireless transmission, and environmental logging. It achieves reliable detection accuracy with low power consumption, offering an affordable, scalable solution for research, aquaculture, and environmental conservation.

---

## Introduction
Microscopic analysis is essential in marine ecosystem studies, water quality monitoring, and microorganism distribution analysis. Traditional microscopy requires manual observation, chemical preservation (formaldehyde, ethanol), and expert knowledge, making it time-consuming and prone to human error.

Advancements in embedded AI enable compact hardware to perform real-time microscopic interpretation. MicroPi, built on Raspberry Pi 4B, captures images, preprocesses them using OpenCV, and uses YOLOv8-SEG for species identification and segmentation. It includes watershed-based segmentation for overlapping organisms.

The solution offers affordability, portability, and high efficiency for researchers, students, and environmental monitoring authorities.

---

## Methodology

### 1. Image Acquisition
- Pi camera mounted with precision objective lens  
- LED illumination for high-contrast imaging  
- Vibration-free mechanical frame  

### 2. Image Preprocessing (OpenCV)
- Gaussian blur  
- Contrast enhancement  
- Color normalization  
- Cropping and ROI extraction  
- Resolution optimization  

### 3. AI Model – YOLOv8-SEG (Quantized)
- Pixel-wise segmentation  
- Multi-organism detection  
- Classification at genus/species level  
- INT8/FP16 quantization for faster inference  
- Reduced input image size  

### 4. Overlap Handling & Mask Refinement
- YOLOv8-SEG initial masks  
- Morphological processing  
- Watershed segmentation  
- Extraction of spatial features: area, perimeter, centroid, contours  

### 5. Storage & Reporting
- SQLite database: counts, labels, timestamps, images  
- CSV export  
- Real-time UI using Flask or Tkinter  

---

## Architecture

### Hardware Architecture
- Raspberry Pi 4B (8GB RAM)  
- Pi camera + objective lens  
- LED illumination  
- Pi touchscreen  
- MicroSD storage  
- Cooling system  

### Software Architecture
- Raspberry Pi OS  
- Python  
- OpenCV  
- YOLOv8-SEG  
- SQLite  
- Tkinter/Flask UI  
- CSV export utilities  

### Architecture Workflow
1. Image Capture  
2. Preprocessing  
3. AI Model Loading  
4. Segmentation & Classification  
5. Overlap Resolution  
6. Data Logging  
7. Display on TFT Screen  
8. CSV/DB Storage  

---

## Description

### Microscope Setup
- Pi camera integrated with tube lens and objective  
- Adjustable focusing knob  
- Stable optical alignment  
- Portable, compact design  

### AI Processing Flow
1. Image → OpenCV preprocessing  
2. YOLOv8-SEG segmentation  
3. Mask extraction for counting  
4. Watershed-based overlap resolution  
5. Bounding masks and labels displayed in real-time  

---

## Key Features
- Handles dense & overlapping microscopic samples  
- Real-time detection & segmentation  
- Works fully offline  
- Low-cost but accurate scientific tool  
- Ideal for marine researchers, aquaculture labs, and colleges  

---

## Applications
- **Marine & Aquaculture:** phytoplankton quantification, bloom monitoring  
- **Environment:** water quality analysis, pollution assessment  
- **Healthcare:** microbial screening  
- **Agriculture:** soil microbe detection  
- **Education & Research:** digital microscope for laboratories  

---

## Results and Discussion

### Sample Output Highlights
- Segments multiple organisms in a single frame  
- Handles 30%–40% overlapping organisms using watershed  
- Produces labeled masks, contours, organism counts  
- Real-time visualization on Raspberry Pi after model optimization  

### Performance Summary

| Parameter | Result |
|----------|--------|
| Inference Time | 150–250 ms/frame (quantized) |
| Segmentation Accuracy | High mask precision |
| Overlap Handling | Effective separation |
| Storage | Lightweight SQLite DB |

### Discussion
MicroPi demonstrates strong performance despite using low-cost hardware. Quantization greatly improves inference speed without major accuracy loss. It enables microscopy without advanced laboratory infrastructure.

---

## Conclusion
MicroPi overcomes the limitations of traditional microscopy by offering a portable, automated AI system for marine organism analysis. Its real-time segmentation, classification, and offline functionality make it suitable for marine biology, environmental monitoring, aquaculture, and educational applications.

Future improvements—such as automated sample preparation—can transform MicroPi into a fully autonomous micro-analysis platform.

---

## References
1. https://pmc.ncbi.nlm.nih.gov/articles/PMC7376023/  
2. https://arxiv.org/pdf/2108.05258  
3. https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1954-8  
4. https://pmc.ncbi.nlm.nih.gov/articles/PMC8478609/  
5. https://aslopubs.onlinelibrary.wiley.com/doi/10.1002/lol2.10392  

Files like `.gitkeep` are present in empty directories so they are tracked by Git.
