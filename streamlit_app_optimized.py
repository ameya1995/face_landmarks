import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image
import io
import zipfile
import gc
from functools import lru_cache

# Page configuration
st.set_page_config(
    page_title="Facial Landmarks Analysis",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize MediaPipe Face Mesh only once and cache it
@st.cache_resource
def get_face_mesh_model():
    """Initialize and cache MediaPipe Face Mesh model"""
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1,
        refine_landmarks=False,  # Disable refinement for better performance
        min_detection_confidence=0.5
    )

# Cache mathematical calculations
@lru_cache(maxsize=128)
def calculate_distance_cached(x1, y1, x2, y2):
    """Cached distance calculation"""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def calculate_distance(point1, point2):
    return calculate_distance_cached(point1.x, point1.y, point2.x, point2.y)

def calculate_horizontal_distance(point1, point2):
    """Calculate the horizontal distance between two points."""
    return abs(point1.x - point2.x)

@lru_cache(maxsize=64)
def calculate_midpoint_cached(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2

def calculate_midpoint(point1, point2):
    x, y = calculate_midpoint_cached(point1.x, point1.y, point2.x, point2.y)
    return {'x': x, 'y': y}

def calculate_custom_30_x_left(point1, point2):
    return {'x': point1.x + ((point2.x - point1.x) / 4), 'y': (point1.y + point2.y) / 2}

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

def calculate_percentage_width(landmarks, point1, point2, total_width):
    """Calculate the width between two points as a percentage of total width."""
    width = calculate_horizontal_distance(landmarks[point1], landmarks[point2])
    return (width / total_width) * 100

def draw_point(image, point_or_landmark, color=(0, 0, 255), radius=3, text=None):
    """Draws a circle at the given point or landmark with reduced radius for performance."""
    height, width, _ = image.shape

    if isinstance(point_or_landmark, dict):
        x = int(point_or_landmark['x'] * width)
        y = int(point_or_landmark['y'] * height)
    else:
        x = int(point_or_landmark.x * width)
        y = int(point_or_landmark.y * height)

    cv2.circle(image, (x, y), radius, color, -1)

    if text is not None:
        cv2.putText(
            image, text, (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

def draw_line(image, landmark1, landmark2, color=(255, 0, 0), thickness=1):
    """Draw a line between two points on the image with reduced thickness."""
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

def optimize_image_size(image_array, max_dimension=800):
    """Resize image if it's too large to reduce memory usage"""
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
def process_image_cached(image_bytes, max_size=800):
    """Process a single image with caching and memory optimization."""
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Optimize image size
        image_array = optimize_image_size(image_array, max_size)
        
        # Get cached face mesh model
        face_mesh = get_face_mesh_model()
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        # Create separate copies for different annotations
        points_image = image_array.copy()
        lines_image = image_array.copy()
        analysis_data = {}
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Compute midpoints
                midpoint_13_14 = calculate_midpoint(landmarks[13], landmarks[14])
                midpoint_57_61 = calculate_custom_30_x_right(landmarks[57], landmarks[61])
                
                # Calculate all ratios efficiently
                ratios = {}
                
                # Pre-calculate common distances to avoid redundant calculations
                dist_2_152 = calculate_distance(landmarks[2], landmarks[152])
                dist_8_2 = calculate_distance(landmarks[8], landmarks[2])
                dist_219_439 = calculate_distance(landmarks[219], landmarks[439])
                
                dist_8_mid13_14 = calculate_distance_from_midpoint(landmarks[8], midpoint_13_14)
                dist_2_mid13_14 = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14)
                dist_mid13_14_152 = calculate_distance_from_midpoint(midpoint_13_14, landmarks[152])
                dist_mid57_61_291 = calculate_distance_from_midpoint(midpoint_57_61, landmarks[291])
                
                ratios['ratio_1'] = dist_8_mid13_14 / dist_2_152
                ratios['ratio_2'] = dist_8_2 / dist_2_152
                ratios['ratio_3'] = dist_2_mid13_14 / dist_2_152
                ratios['ratio_4'] = dist_mid13_14_152 / dist_2_152
                ratios['ratio_5'] = dist_2_mid13_14 / dist_mid13_14_152
                ratios['ratio_6'] = calculate_distance_from_midpoint(landmarks[0], midpoint_13_14) / dist_2_mid13_14
                ratios['ratio_7'] = calculate_distance_from_midpoint(landmarks[0], midpoint_13_14) / calculate_distance_from_midpoint(midpoint_13_14, landmarks[17])
                ratios['ratio_8'] = dist_219_439 / dist_mid57_61_291
                ratios['ratio_9'] = dist_mid57_61_291 / calculate_distance(landmarks[130], landmarks[263])
                ratios['ratio_10'] = dist_219_439 / dist_8_2
                ratios['ratio_11'] = dist_2_mid13_14 / dist_mid57_61_291
                ratios['ratio_12'] = dist_2_152 / dist_mid57_61_291
                
                analysis_data = {
                    'ratios': ratios,
                    'landmarks_count': len(landmarks),
                    'image_dimensions': image_array.shape[:2]
                }
                
                # Draw points and lines with reduced complexity
                key_landmarks = [0, 2, 8, 13, 14, 17, 57, 61, 130, 152, 219, 263, 291, 439]
                
                for i in key_landmarks:
                    draw_point(points_image, landmarks[i], radius=2)
                
                # Draw key lines
                key_connections = [
                    (2, 152), (8, 2), (219, 439), (130, 263)
                ]
                
                for p1, p2 in key_connections:
                    draw_line(lines_image, landmarks[p1], landmarks[p2])
                
                # Draw midpoint connections
                draw_line(lines_image, landmarks[13], landmarks[14])
                draw_point(lines_image, midpoint_13_14, color=(0, 255, 0), radius=2)
                draw_point(lines_image, midpoint_57_61, color=(0, 255, 0), radius=2)
        
        # Clean up memory
        del image_rgb
        gc.collect()
        
        return points_image, lines_image, analysis_data
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None

# Custom CSS for better styling (simplified)
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1.5rem; }
    .metric-container { background-color: #f0f2f6; padding: 0.8rem; border-radius: 0.4rem; margin: 0.3rem 0; }
    .stAlert { margin-top: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# Main app header
st.markdown('<h1 class="main-header">👤 Facial Landmarks Analysis</h1>', unsafe_allow_html=True)

# Sidebar for options
with st.sidebar:
    st.header("⚙️ Settings")
    max_image_size = st.slider("Max Image Size (px)", 400, 1200, 800, 100, 
                              help="Smaller size = faster processing, less memory usage")
    show_ratios = st.checkbox("Show Detailed Ratios", value=True)
    batch_size = st.slider("Batch Processing Size", 1, 5, 3, 
                          help="Number of images to process simultaneously")

# Main content
tab1, tab2 = st.tabs(["📊 Single Image Analysis", "📁 Batch Processing"])

with tab1:
    st.subheader("Upload an Image for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'], 
        help="Upload a clear facial image for landmark analysis"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            # Read and process the image
            image_bytes = uploaded_file.read()
            points_img, lines_img, analysis = process_image_cached(image_bytes, max_image_size)
            
            if points_img is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔴 Key Landmarks")
                    st.image(points_img, channels="BGR", use_column_width=True)
                
                with col2:
                    st.subheader("📏 Measurement Lines")
                    st.image(lines_img, channels="BGR", use_column_width=True)
                
                if show_ratios and analysis:
                    st.subheader("📊 Analysis Results")
                    
                    # Display ratios in a more compact format
                    ratios_df = pd.DataFrame([analysis['ratios']]).T
                    ratios_df.columns = ['Value']
                    ratios_df.index.name = 'Ratio'
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(ratios_df.head(6), use_container_width=True)
                    with col2:
                        st.dataframe(ratios_df.tail(6), use_container_width=True)
                    
                    st.metric("Image Dimensions", f"{analysis['image_dimensions'][1]}x{analysis['image_dimensions'][0]}")
                    st.metric("Landmarks Detected", analysis['landmarks_count'])

with tab2:
    st.subheader("Batch Image Processing")
    st.info("⚡ Optimized for memory efficiency - processes images in smaller batches")
    
    uploaded_files = st.file_uploader(
        "Choose multiple image files", 
        type=['png', 'jpg', 'jpeg'], 
        accept_multiple_files=True,
        help=f"Upload up to {batch_size} images at once for efficient processing"
    )
    
    if uploaded_files:
        # Limit batch size for memory management
        files_to_process = uploaded_files[:batch_size]
        
        if len(uploaded_files) > batch_size:
            st.warning(f"Processing first {batch_size} images. Upload remaining images separately.")
        
        if st.button("🚀 Process Batch", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for idx, file in enumerate(files_to_process):
                status_text.text(f"Processing {file.name}...")
                
                # Process each image
                image_bytes = file.read()
                points_img, lines_img, analysis = process_image_cached(image_bytes, max_image_size)
                
                if points_img is not None:
                    results.append({
                        'filename': file.name,
                        'points_image': points_img,
                        'lines_image': lines_img,
                        'analysis': analysis
                    })
                
                progress_bar.progress((idx + 1) / len(files_to_process))
                
                # Clear cache periodically to manage memory
                if (idx + 1) % 2 == 0:
                    gc.collect()
            
            status_text.text("✅ Processing complete!")
            
            # Display results
            if results:
                st.subheader(f"📋 Results ({len(results)} images processed)")
                
                # Create download packages
                points_zip = io.BytesIO()
                lines_zip = io.BytesIO()
                
                with zipfile.ZipFile(points_zip, 'w') as pz, zipfile.ZipFile(lines_zip, 'w') as lz:
                    for result in results:
                        filename_base = result['filename'].rsplit('.', 1)[0]
                        
                        # Encode images
                        _, points_encoded = cv2.imencode('.jpg', result['points_image'], [cv2.IMWRITE_JPEG_QUALITY, 85])
                        _, lines_encoded = cv2.imencode('.jpg', result['lines_image'], [cv2.IMWRITE_JPEG_QUALITY, 85])
                        
                        pz.writestr(f"{filename_base}_points.jpg", points_encoded.tobytes())
                        lz.writestr(f"{filename_base}_lines.jpg", lines_encoded.tobytes())
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📍 Download Points Images (ZIP)",
                        data=points_zip.getvalue(),
                        file_name="landmarks_points.zip",
                        mime="application/zip"
                    )
                
                with col2:
                    st.download_button(
                        "📏 Download Lines Images (ZIP)",
                        data=lines_zip.getvalue(),
                        file_name="landmarks_lines.zip",
                        mime="application/zip"
                    )
                
                # Summary statistics
                if show_ratios:
                    st.subheader("📊 Batch Analysis Summary")
                    
                    all_ratios = []
                    for result in results:
                        if result['analysis']:
                            ratios_row = result['analysis']['ratios'].copy()
                            ratios_row['filename'] = result['filename']
                            all_ratios.append(ratios_row)
                    
                    if all_ratios:
                        summary_df = pd.DataFrame(all_ratios)
                        summary_df = summary_df.set_index('filename')
                        
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Download CSV
                        csv_buffer = io.StringIO()
                        summary_df.to_csv(csv_buffer)
                        
                        st.download_button(
                            "📊 Download Analysis CSV",
                            data=csv_buffer.getvalue(),
                            file_name="facial_analysis_results.csv",
                            mime="text/csv"
                        )

# Footer
st.markdown("---")
st.markdown("💡 **Tips for better performance:**")
st.markdown("- Use smaller image sizes for faster processing")
st.markdown("- Process images in smaller batches")
st.markdown("- Clear browser cache if the app becomes slow")
