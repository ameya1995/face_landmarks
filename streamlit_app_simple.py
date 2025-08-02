import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image
import io
import zipfile

# Page configuration
st.set_page_config(
    page_title="Facial Landmarks Analysis",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize MediaPipe Face Mesh only once
@st.cache_resource
def get_face_mesh_model():
    """Initialize and cache MediaPipe Face Mesh model"""
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

def calculate_horizontal_distance(point1, point2):
    return abs(point1.x - point2.x)

def calculate_midpoint(point1, point2):
    return {'x': (point1.x + point2.x) / 2, 'y': (point1.y + point2.y) / 2}

def calculate_custom_30_x_right(point1, point2):
    return {'x': point1.x + (3 * (point2.x - point1.x) / 4), 'y': (point1.y + point2.y) / 2}

def calculate_distance_from_midpoint(point_or_midpoint1, point_or_midpoint2):
    if isinstance(point_or_midpoint1, dict):
        point1 = np.array([point_or_midpoint1['x'], point_or_midpoint1['y']], dtype=np.float32)
    else:
        point1 = np.array([point_or_midpoint1.x, point_or_midpoint1.y], dtype=np.float32)

    if isinstance(point_or_midpoint2, dict):
        point2 = np.array([point_or_midpoint2['x'], point_or_midpoint2['y']], dtype=np.float32)
    else:
        point2 = np.array([point_or_midpoint2.x, point_or_midpoint2.y], dtype=np.float32)

    return np.linalg.norm(point1 - point2)

def draw_point(image, point_or_landmark, color=(0, 0, 255), radius=3):
    height, width, _ = image.shape
    if isinstance(point_or_landmark, dict):
        x = int(point_or_landmark['x'] * width)
        y = int(point_or_landmark['y'] * height)
    else:
        x = int(point_or_landmark.x * width)
        y = int(point_or_landmark.y * height)
    cv2.circle(image, (x, y), radius, color, -1)

def draw_line(image, landmark1, landmark2, color=(255, 0, 0), thickness=1):
    height, width, _ = image.shape
    if isinstance(landmark1, dict):
        start_point = (int(landmark1['x'] * width), int(landmark1['y'] * height))
    else:
        start_point = (int(landmark1.x * width), int(landmark1.y * height))
    
    if isinstance(landmark2, dict):
        end_point = (int(landmark2['x'] * width), int(landmark2['y'] * height))
    else:
        end_point = (int(landmark2.x * width), int(landmark2.y * height))

    cv2.line(image, start_point, end_point, color, thickness)

def resize_image_if_needed(image_array, max_dimension=800):
    height, width = image_array.shape[:2]
    if max(height, width) > max_dimension:
        if height > width:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        else:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        return cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image_array

@st.cache_data
def process_image_simple(image_bytes, max_size=800):
    """Simplified image processing with caching"""
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize if needed
        image_array = resize_image_if_needed(image_array, max_size)
        
        # Get face mesh model
        face_mesh = get_face_mesh_model()
        
        # Process image
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        # Create copies for annotation
        points_image = image_array.copy()
        lines_image = image_array.copy()
        analysis_data = {}
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Calculate key measurements
                midpoint_13_14 = calculate_midpoint(landmarks[13], landmarks[14])
                
                # Calculate basic ratios
                dist_2_152 = calculate_distance(landmarks[2], landmarks[152])
                dist_8_2 = calculate_distance(landmarks[8], landmarks[2])
                
                ratios = {
                    'ratio_1': calculate_distance_from_midpoint(landmarks[8], midpoint_13_14) / dist_2_152,
                    'ratio_2': dist_8_2 / dist_2_152,
                    'ratio_3': calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / dist_2_152,
                    'ratio_4': calculate_distance_from_midpoint(midpoint_13_14, landmarks[152]) / dist_2_152,
                }
                
                analysis_data = {
                    'ratios': ratios,
                    'landmarks_count': len(landmarks),
                    'image_dimensions': image_array.shape[:2]
                }
                
                # Draw key landmarks
                key_points = [0, 2, 8, 13, 14, 17, 57, 61, 130, 152, 219, 263, 291, 439]
                for i in key_points:
                    draw_point(points_image, landmarks[i])
                
                # Draw key lines
                draw_line(lines_image, landmarks[2], landmarks[152])
                draw_line(lines_image, landmarks[8], landmarks[2])
                draw_line(lines_image, landmarks[13], landmarks[14])
                draw_point(lines_image, midpoint_13_14, color=(0, 255, 0))
        
        return points_image, lines_image, analysis_data
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None

# Main app
st.title("👤 Facial Landmarks Analysis (Lightweight)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_size = st.slider("Max Image Size", 400, 1000, 600, 50)
    st.info("Smaller size = faster processing")

# File upload
uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    with st.spinner("Processing..."):
        image_bytes = uploaded_file.read()
        points_img, lines_img, analysis = process_image_simple(image_bytes, max_size)
        
        if points_img is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔴 Key Points")
                st.image(points_img, channels="BGR", use_column_width=True)
            
            with col2:
                st.subheader("📏 Measurements")
                st.image(lines_img, channels="BGR", use_column_width=True)
            
            if analysis:
                st.subheader("📊 Results")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Landmarks", analysis['landmarks_count'])
                with col2:
                    st.metric("Image Size", f"{analysis['image_dimensions'][1]}x{analysis['image_dimensions'][0]}")
                with col3:
                    st.metric("Ratios Calculated", len(analysis['ratios']))
                
                # Show ratios
                if st.checkbox("Show Detailed Ratios"):
                    ratios_df = pd.DataFrame([analysis['ratios']]).T
                    ratios_df.columns = ['Value']
                    st.dataframe(ratios_df, use_container_width=True)

# Batch processing
st.subheader("📁 Batch Processing")
uploaded_files = st.file_uploader("Multiple images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files and st.button("Process All"):
    progress = st.progress(0)
    results = []
    
    for i, file in enumerate(uploaded_files[:3]):  # Limit to 3 for memory
        image_bytes = file.read()
        points_img, lines_img, analysis = process_image_simple(image_bytes, max_size)
        
        if points_img is not None:
            results.append({
                'name': file.name,
                'points': points_img,
                'lines': lines_img,
                'data': analysis
            })
        
        progress.progress((i + 1) / min(len(uploaded_files), 3))
    
    if results:
        st.success(f"Processed {len(results)} images!")
        
        # Create download package
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            for result in results:
                name_base = result['name'].rsplit('.', 1)[0]
                _, encoded = cv2.imencode('.jpg', result['points'])
                zf.writestr(f"{name_base}_landmarks.jpg", encoded.tobytes())
        
        st.download_button(
            "📦 Download Results",
            data=zip_buffer.getvalue(),
            file_name="landmarks_results.zip",
            mime="application/zip"
        )

st.markdown("---")
st.markdown("💡 **Tip**: Use smaller image sizes for faster processing on free hosting tiers.")
