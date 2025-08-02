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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        ma                        st.download_button(
                            label="📏 Download Lines Images (ZIP)",
                            data=zip_buffer_lines.getvalue(),
                            file_name="lines_images.zip",
                            mime="application/zip"
                        )ottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function definitions (from your original code)
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

def calculate_horizontal_distance(point1, point2):
    """Calculate the horizontal distance between two points."""
    return abs(point1.x - point2.x)

def calculate_midpoint(point1, point2):
    return {'x': (point1.x + point2.x) / 2, 'y': (point1.y + point2.y) / 2}

def calculate_custom_30_x_left(point1, point2):
    return {'x': point1.x + ((point2.x - point1.x) / 4), 'y': (point1.y + point2.y) / 2}

def calculate_custom_30_x_right(point1, point2):
    return {'x': point1.x + (3 * (point2.x - point1.x) / 4), 'y': (point1.y + point2.y) / 2}

def calculate_distance_from_midpoint(point_or_midpoint1, point_or_midpoint2):
    if isinstance(point_or_midpoint1, dict):
        point1 = np.array([point_or_midpoint1['x'], point_or_midpoint1['y']])
    else:
        point1 = np.array([point_or_midpoint1.x, point_or_midpoint1.y])

    if isinstance(point_or_midpoint2, dict):
        point2 = np.array([point_or_midpoint2['x'], point_or_midpoint2['y']])
    else:
        point2 = np.array([point_or_midpoint2.x, point_or_midpoint2.y])

    return np.linalg.norm(point1 - point2)

def calculate_percentage_width(landmarks, point1, point2, total_width):
    """Calculate the width between two points as a percentage of total width."""
    width = calculate_horizontal_distance(landmarks[point1], landmarks[point2])
    return (width / total_width) * 100

def draw_point(image, point_or_landmark, color=(0, 0, 255), radius=5, text=None):
    """Draws a circle at the given point or landmark."""
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
            image, text, (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

def draw_line(image, landmark1, landmark2, color=(255, 0, 0), thickness=2):
    """Draw a line between two points on the image."""
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

def process_image(image_array):
    """Process a single image and return analysis results."""
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    
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
            
            # Calculate all ratios
            ratios = {}
            ratios['ratio_1'] = calculate_distance_from_midpoint(landmarks[8], midpoint_13_14) / calculate_distance(landmarks[2], landmarks[152])
            ratios['ratio_2'] = calculate_distance(landmarks[8], landmarks[2]) / calculate_distance(landmarks[2], landmarks[152])
            ratios['ratio_3'] = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance(landmarks[2], landmarks[152])
            ratios['ratio_4'] = calculate_distance_from_midpoint(midpoint_13_14, landmarks[152]) / calculate_distance(landmarks[2], landmarks[152])
            ratios['ratio_5'] = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance_from_midpoint(midpoint_13_14, landmarks[152])
            ratios['ratio_6'] = calculate_distance_from_midpoint(landmarks[0], midpoint_13_14) / calculate_distance_from_midpoint(landmarks[2], midpoint_13_14)
            ratios['ratio_7'] = calculate_distance_from_midpoint(landmarks[0], midpoint_13_14) / calculate_distance_from_midpoint(midpoint_13_14, landmarks[17])
            ratios['ratio_8'] = calculate_distance(landmarks[219], landmarks[439]) / calculate_distance_from_midpoint(midpoint_57_61, landmarks[291])
            ratios['ratio_9'] = calculate_distance_from_midpoint(midpoint_57_61, landmarks[291]) / calculate_distance(landmarks[130], landmarks[263])
            ratios['ratio_10'] = calculate_distance(landmarks[219], landmarks[439]) / calculate_distance(landmarks[8], landmarks[2])
            ratios['ratio_11'] = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance_from_midpoint(midpoint_57_61, landmarks[291])
            ratios['ratio_12'] = calculate_distance(landmarks[2], landmarks[152]) / calculate_distance_from_midpoint(midpoint_57_61, landmarks[291])
            ratios['ratio_13'] = calculate_distance(landmarks[8], landmarks[152]) / calculate_distance(landmarks[234], landmarks[454])
            ratios['ratio_14'] = calculate_distance(landmarks[2], landmarks[0]) / calculate_distance_from_midpoint(landmarks[2], midpoint_13_14)
            ratios['ratio_15'] = calculate_distance_from_midpoint(landmarks[0], midpoint_13_14) / calculate_distance(landmarks[2], landmarks[0])
            ratios['ratio_16'] = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance(landmarks[2], landmarks[0])
            ratios['ratio_17'] = calculate_distance_from_midpoint(landmarks[8], midpoint_13_14) / calculate_distance(landmarks[130], landmarks[263])
            ratios['ratio_18'] = calculate_distance(landmarks[155], landmarks[463]) / calculate_distance(landmarks[219], landmarks[439])
            ratios['ratio_19'] = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance_from_midpoint(landmarks[8], midpoint_13_14)
            ratios['ratio_20'] = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance(landmarks[8], landmarks[2])
            ratios['ratio_21'] = calculate_distance(landmarks[155], landmarks[463]) / calculate_distance(landmarks[130], landmarks[263])
            ratios['ratio_22'] = calculate_distance(landmarks[463], landmarks[263]) / calculate_distance(landmarks[155], landmarks[463])
            
            # Calculate width percentages
            total_width = calculate_distance(landmarks[127], landmarks[447])
            width_percentages = {
                'percentage_width_130_133': calculate_percentage_width(landmarks, 130, 133, total_width),
                'percentage_width_133_463': calculate_percentage_width(landmarks, 133, 463, total_width),
                'percentage_width_463_263': calculate_percentage_width(landmarks, 463, 263, total_width),
            }
            
            analysis_data = {**ratios, **width_percentages}
            
            # Create points annotation
            key_points = [
                (2, "Sn"), (8, "N'"), (13, "St"), (17, "Li"),
                (152, "Me'"), (219, "Alr"), (439, "All"),
                (130, "Exr"), (263, "Exl"), (234, "Zyr"), (454, "Zyl"),
                (155, "Enr"), (463, "Enl"), (291, "Chl")
            ]
            for (idx, label) in key_points:
                draw_point(points_image, landmarks[idx], color=(0, 0, 255), radius=12, text=label)
            
            # Draw midpoints
            draw_point(points_image, midpoint_57_61, color=(0, 255, 255), radius=12, text="Chr")
            
            # Create lines annotation (optimized to remove overlaps, keeping longer lines)
            # Main vertical facial measurements
            draw_line(lines_image, landmarks[2], landmarks[152], (255, 0, 0), 5)  # Sn to Me' (full face height)
            draw_line(lines_image, landmarks[8], landmarks[152], (0, 255, 0), 5)  # N' to Me' (alternative height)
            draw_line(lines_image, landmarks[0], landmarks[17], (255, 0, 0), 5)   # Upper to lower face
            
            # Horizontal facial measurements
            draw_line(lines_image, landmarks[219], landmarks[439], (255, 0, 0), 5)  # Nostril width
            draw_line(lines_image, landmarks[130], landmarks[263], (255, 0, 0), 5)  # Eye width
            draw_line(lines_image, landmarks[155], landmarks[463], (255, 0, 0), 5)  # Inner eye width
            draw_line(lines_image, landmarks[234], landmarks[454], (255, 0, 0), 5)  # Cheek width
            
            # Key vertical reference lines
            draw_line(lines_image, landmarks[8], midpoint_13_14, (0, 0, 255), 5)   # N' to midpoint
            draw_line(lines_image, midpoint_57_61, landmarks[291], (255, 0, 0), 5)  # Midpoint to Chl
    
    face_mesh.close()
    
    # Return original image, points image, lines image, and analysis data
    return image_array, points_image, lines_image, analysis_data

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Facial Landmarks Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Upload Options")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Input Method:",
        ["Single Image", "Multiple Images", "Camera Capture"],
        help="Choose whether to upload images or capture from camera"
    )
    
    # Main content area
    if input_method == "Single Image":
        st.subheader("📤 Upload Single Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'JPG'],
            help="Upload a clear facial image for analysis"
        )
        
        if uploaded_file is not None:
            # Display original image and processed images
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📷 Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            
            # Process image
            image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            with st.spinner("🔄 Processing image..."):
                original_image, points_image, lines_image, analysis_data = process_image(image_array)
            
            with col2:
                st.subheader(" Key Points")
                points_image_rgb = cv2.cvtColor(points_image, cv2.COLOR_BGR2RGB)
                st.image(points_image_rgb, use_container_width=True)
            
            with col3:
                st.subheader("📏 Measurement Lines")
                lines_image_rgb = cv2.cvtColor(lines_image, cv2.COLOR_BGR2RGB)
                st.image(lines_image_rgb, use_container_width=True)
            
            # Display analysis results
            if analysis_data:
                st.subheader("📊 Analysis Results")
                
                # Ratios section
                with st.expander("📏 Facial Ratios", expanded=True):
                    ratio_cols = st.columns(4)
                    ratio_keys = [k for k in analysis_data.keys() if k.startswith('ratio_')]
                    
                    for i, ratio_key in enumerate(ratio_keys):
                        with ratio_cols[i % 4]:
                            st.metric(
                                label=f"Ratio {ratio_key.split('_')[1]}",
                                value=f"{analysis_data[ratio_key]:.3f}",
                                help=f"Calculated {ratio_key.replace('_', ' ')}"
                            )
                
                # Width percentages section
                with st.expander("📐 Width Proportions"):
                    width_cols = st.columns(3)
                    width_keys = [k for k in analysis_data.keys() if k.startswith('percentage_')]
                    
                    for i, width_key in enumerate(width_keys):
                        with width_cols[i]:
                            st.metric(
                                label=f"Proportion {chr(65+i)}",  # A, B, C
                                value=f"{analysis_data[width_key]:.2f}%",
                                help=f"Width percentage for segment {chr(65+i)}"
                            )
                
                # Download results
                st.subheader("💾 Download Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download points image
                    points_pil = Image.fromarray(points_image_rgb)
                    points_buffer = io.BytesIO()
                    points_pil.save(points_buffer, format='PNG')
                    points_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Points Image",
                        data=points_buffer,
                        file_name=f"points_{uploaded_file.name}",
                        mime="image/png"
                    )
                
                with col2:
                    # Download lines image
                    lines_pil = Image.fromarray(lines_image_rgb)
                    lines_buffer = io.BytesIO()
                    lines_pil.save(lines_buffer, format='PNG')
                    lines_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Lines Image",
                        data=lines_buffer,
                        file_name=f"lines_{uploaded_file.name}",
                        mime="image/png"
                    )
                
                with col3:
                    # Download analysis data as CSV
                    df = pd.DataFrame([analysis_data])
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    st.download_button(
                        label="📊 Download Analysis Data (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=f"analysis_{uploaded_file.name.split('.')[0]}.csv",
                        mime="text/csv"
                    )
    
    elif input_method == "Camera Capture":
        st.subheader("📸 Camera Capture")
        
        # Camera capture interface
        camera_image = st.camera_input("Take a photo for facial analysis")
        
        if camera_image is not None:
            # Display captured image and processed images
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📷 Captured Image")
                image = Image.open(camera_image)
                st.image(image, use_container_width=True)
            
            # Process image
            image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            with st.spinner("🔄 Processing image..."):
                original_image, points_image, lines_image, analysis_data = process_image(image_array)
            
            with col2:
                st.subheader("🔴 Key Points")
                points_image_rgb = cv2.cvtColor(points_image, cv2.COLOR_BGR2RGB)
                st.image(points_image_rgb, use_container_width=True)
            
            with col3:
                st.subheader("📏 Measurement Lines")
                lines_image_rgb = cv2.cvtColor(lines_image, cv2.COLOR_BGR2RGB)
                st.image(lines_image_rgb, use_container_width=True)
            
            # Display analysis results
            if analysis_data:
                st.subheader("📊 Analysis Results")
                
                # Ratios section
                with st.expander("📏 Facial Ratios", expanded=True):
                    ratio_cols = st.columns(4)
                    ratio_keys = [k for k in analysis_data.keys() if k.startswith('ratio_')]
                    
                    for i, ratio_key in enumerate(ratio_keys):
                        with ratio_cols[i % 4]:
                            st.metric(
                                label=f"Ratio {ratio_key.split('_')[1]}",
                                value=f"{analysis_data[ratio_key]:.3f}",
                                help=f"Calculated {ratio_key.replace('_', ' ')}"
                            )
                
                # Width percentages section
                with st.expander("📐 Width Proportions"):
                    width_cols = st.columns(3)
                    width_keys = [k for k in analysis_data.keys() if k.startswith('percentage_')]
                    
                    for i, width_key in enumerate(width_keys):
                        with width_cols[i]:
                            st.metric(
                                label=f"Proportion {chr(65+i)}",  # A, B, C
                                value=f"{analysis_data[width_key]:.2f}%",
                                help=f"Width percentage for segment {chr(65+i)}"
                            )
                
                # Download results
                st.subheader("💾 Download Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download points image
                    points_pil = Image.fromarray(points_image_rgb)
                    points_buffer = io.BytesIO()
                    points_pil.save(points_buffer, format='PNG')
                    points_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Points Image",
                        data=points_buffer,
                        file_name="camera_points_image.png",
                        mime="image/png"
                    )
                
                with col2:
                    # Download lines image
                    lines_pil = Image.fromarray(lines_image_rgb)
                    lines_buffer = io.BytesIO()
                    lines_pil.save(lines_buffer, format='PNG')
                    lines_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Lines Image",
                        data=lines_buffer,
                        file_name="camera_lines_image.png",
                        mime="image/png"
                    )
                
                with col3:
                    # Download analysis data as CSV
                    df = pd.DataFrame([analysis_data])
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    st.download_button(
                        label="📊 Download Analysis Data (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name="camera_analysis.csv",
                        mime="text/csv"
                    )
    
    else:  # Multiple Images
        st.subheader("📤 Upload Multiple Images")
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg', 'JPG'],
            accept_multiple_files=True,
            help="Upload multiple facial images for batch analysis"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} images uploaded successfully!")
            
            if st.button("🚀 Process All Images", type="primary"):
                results_data = []
                processed_images = []
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                    
                    # Process each image
                    image = Image.open(uploaded_file)
                    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    original_image, points_image, lines_image, analysis_data = process_image(image_array)
                    
                    if analysis_data:
                        analysis_data['Image'] = uploaded_file.name
                        results_data.append(analysis_data)
                        
                        # Store all processed images
                        points_image_rgb = cv2.cvtColor(points_image, cv2.COLOR_BGR2RGB)
                        lines_image_rgb = cv2.cvtColor(lines_image, cv2.COLOR_BGR2RGB)
                        
                        processed_images.append({
                            'filename': uploaded_file.name,
                            'points': points_image_rgb,
                            'lines': lines_image_rgb
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ Processing complete!")
                
                # Display results
                if results_data:
                    st.subheader("📊 Batch Analysis Results")
                    
                    # Create DataFrame
                    results_df = pd.DataFrame(results_data)
                    
                    # Display summary statistics
                    with st.expander("📈 Summary Statistics", expanded=True):
                        st.dataframe(results_df.describe())
                    
                    # Display detailed results
                    with st.expander("🔍 Detailed Results"):
                        st.dataframe(results_df)
                    
                    # Display processed images gallery
                    st.subheader("🖼️ Processed Images Gallery")
                    
                    # Create tabs for different image types
                    tab1, tab2 = st.tabs(["🔴 Points Images", "📏 Lines Images"])
                    
                    with tab1:
                        cols = st.columns(3)
                        for i, img_data in enumerate(processed_images):
                            with cols[i % 3]:
                                st.image(img_data['points'], caption=f"Points - {img_data['filename']}", use_container_width=True)
                    
                    with tab2:
                        cols = st.columns(3)
                        for i, img_data in enumerate(processed_images):
                            with cols[i % 3]:
                                st.image(img_data['lines'], caption=f"Lines - {img_data['filename']}", use_container_width=True)
                    
                    # Download options
                    st.subheader("💾 Download Batch Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Download CSV
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        
                        st.download_button(
                            label="📊 Download Analysis Data (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name="batch_analysis_results.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Create ZIP file with points images
                        zip_buffer_points = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer_points, 'w') as zip_file:
                            for img_data in processed_images:
                                img_pil = Image.fromarray(img_data['points'])
                                img_bytes = io.BytesIO()
                                img_pil.save(img_bytes, format='PNG')
                                zip_file.writestr(f"points_{img_data['filename']}", img_bytes.getvalue())
                        
                        zip_buffer_points.seek(0)
                        st.download_button(
                            label="🔴 Download Points Images (ZIP)",
                            data=zip_buffer_points.getvalue(),
                            file_name="points_images.zip",
                            mime="application/zip"
                        )
                    
                    with col3:
                        # Create ZIP file with lines images
                        zip_buffer_lines = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer_lines, 'w') as zip_file:
                            for img_data in processed_images:
                                img_pil = Image.fromarray(img_data['lines'])
                                img_bytes = io.BytesIO()
                                img_pil.save(img_bytes, format='PNG')
                                zip_file.writestr(f"lines_{img_data['filename']}", img_bytes.getvalue())
                        
                        zip_buffer_lines.seek(0)
                        st.download_button(
                            label="� Download Lines Images (ZIP)",
                            data=zip_buffer_lines.getvalue(),
                            file_name="lines_images.zip",
                            mime="application/zip"
                        )
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        **Facial Landmarks Analysis Application**
        
        This application uses MediaPipe's Face Mesh to detect and analyze facial landmarks, calculating:
        
        - **22 Different Facial Ratios**: Measurements between key facial points
        - **3 Width Proportions**: Facial width divided into three segments (A, B, C)
        - **Visual Annotations**: Key anatomical points and measurement lines
        
        **Input Methods:**
        - **Single Image**: Upload a single image file for analysis
        - **Multiple Images**: Upload multiple images for batch processing
        - **Camera Capture**: Take a photo directly from your webcam for real-time analysis
        
        **Features:**
        - **Original Image**: Shows the uploaded/captured image without modifications
        - **Key Points**: Displays facial anatomical landmarks with labels
        - **Measurement Lines**: Shows the lines used for ratio calculations
        - **Download Options**: Save processed images and analysis data
        
        **Supported formats:** PNG, JPG, JPEG
        """)

if __name__ == "__main__":
    main()
