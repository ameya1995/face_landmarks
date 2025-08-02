# Facial Landmarks Analysis Streamlit App

This Streamlit application provides an interactive interface for analyzing facial landmarks using MediaPipe's Face Mesh technology.

## Features

- **Single Image Analysis**: Upload and analyze individual facial images
- **Batch Processing**: Process multiple images simultaneously
- **Multiple Analysis Types**: 
  - Basic: Landmark detection without visual annotations
  - Points: Key facial points with labels
  - Lines: Measurement lines between points
  - Proportions: Vertical proportion guidelines
- **Comprehensive Metrics**: 22 facial ratios and 3 width proportions
- **Export Options**: Download annotated images and analysis data

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Open your web browser and navigate to the displayed local URL (typically `http://localhost:8501`)

3. Use the sidebar to:
   - Choose your analysis type
   - Select single or multiple image upload

4. Upload your facial images (supported formats: PNG, JPG, JPEG)

5. View the results and download processed images or analysis data

## Analysis Types

### Basic
- Detects facial landmarks without visual annotations
- Provides numerical analysis only

### Points
- Shows key facial anatomical points with labels:
  - Sn (Subnasale), N' (Nasion), St (Stomion), Li (Labrale inferius)
  - Me' (Menton), Alr/All (Alar), Exr/Exl (Exocanthion)
  - Zyr/Zyl (Zygion), Enr/Enl (Endocanthion), Chl (Cheilion)
  - Chr (Christa philtri)

### Lines
- Displays measurement lines between key facial points
- Visual representation of the calculated ratios

### Proportions
- Shows vertical guidelines dividing the face into thirds
- Displays width proportions as percentages

## Output

### Metrics
- **22 Facial Ratios**: Various proportional measurements
- **3 Width Proportions**: Face divided into segments A, B, C

### Downloads
- Annotated images with visual analysis
- CSV files with numerical data
- ZIP files for batch processing results

## File Structure

```
face_landmarks_app/
├── streamlit_app.py          # Main Streamlit application
├── landmarks_v1.py           # Original batch processing script
├── landmarks_point_annotation.py  # Original annotation script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Requirements

- Python 3.7+
- Streamlit
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Pillow

## Notes

- Ensure facial images are clear and well-lit for best results
- The application works best with frontal face views
- Multiple faces in a single image will analyze only the first detected face
