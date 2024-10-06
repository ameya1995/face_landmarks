import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av

# Set the page configuration
st.set_page_config(
    page_title="Face Ratio Analyzer",
    page_icon=":face_with_monocle:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with app information
with st.sidebar:
    st.title("📏 Face Ratio Analyzer")
    st.markdown("""
    Upload an image or use your webcam to analyze facial ratios based on landmarks detected by MediaPipe's Face Mesh.

    The app will display the **annotated image** and a **table of calculated ratios**.
    """)
    st.markdown("---")
    st.markdown("Developed by Ameya")

# Main content
st.header("Upload an Image or Use Webcam to Get Started")

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

def calculate_horizontal_distance(point1, point2):
    """Calculate the horizontal distance between two points."""
    return abs(point1.x - point2.x)

# Function to calculate the midpoint between two points
def calculate_midpoint(point1, point2):
    return {'x': (point1.x + point2.x) / 2, 'y': (point1.y + point2.y) / 2}

# Function to calculate the distance from a point to a midpoint
def calculate_distance_from_midpoint(point_or_midpoint1, point_or_midpoint2):
    # Check if first argument is a midpoint
    if isinstance(point_or_midpoint1, dict):
        point1 = np.array([point_or_midpoint1['x'], point_or_midpoint1['y']])
    else:  # if the first argument is a landmark
        point1 = np.array([point_or_midpoint1.x, point_or_midpoint1.y])

    # Check if second argument is a midpoint
    if isinstance(point_or_midpoint2, dict):
        point2 = np.array([point_or_midpoint2['x'], point_or_midpoint2['y']])
    else:  # if the second argument is a landmark
        point2 = np.array([point_or_midpoint2.x, point_or_midpoint2.y])

    return np.linalg.norm(point1 - point2)

def draw_line(image, landmark1, landmark2, color=(255, 0, 0), thickness=2):
    """Draw a line between two points on the image."""
    height, width, _ = image.shape

    if isinstance(landmark1, dict):  # If the first point is a midpoint
        start_point = (int(landmark1['x'] * width), int(landmark1['y'] * height))
    else:
        start_point = (int(landmark1.x * width), int(landmark1.y * height))

    if isinstance(landmark2, dict):  # If the second point is a midpoint
        end_point = (int(landmark2['x'] * width), int(landmark2['y'] * height))
    else:
        end_point = (int(landmark2.x * width), int(landmark2.y * height))

    cv2.line(image, start_point, end_point, color, thickness)

def draw_vertical_line(image, landmark, landmarks, color=(0, 255, 200), thickness=2):
    """Draw a vertical line passing through a landmark."""
    height, width, _ = image.shape
    x = int(landmark.x * width)

    # Calculate top and bottom with buffer
    top_landmark = landmarks[10]  # Top of the forehead
    bottom_landmark = landmarks[152]  # Bottom of the chin
    face_height = bottom_landmark.y - top_landmark.y
    buffer_ratio = 0.15
    buffer = face_height * buffer_ratio

    top = int((top_landmark.y - buffer) * height)
    bottom = int((bottom_landmark.y + buffer) * height)

    # Ensure top and bottom do not go beyond image boundaries
    top = max(0, top)
    bottom = min(height, bottom)

    cv2.line(image, (x, top), (x, bottom), color, thickness)

def calculate_percentage_width(landmarks, point1, point2, total_width):
    """Calculate the width between two points as a percentage of total width."""
    width = calculate_horizontal_distance(landmarks[point1], landmarks[point2])
    return (width / total_width) * 100

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Option to choose between uploading an image or using webcam
option = st.radio("Select Input Method:", ('Upload Image', 'Use Webcam'))

if option == 'Upload Image':
    # Create a file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        process_image = True
    else:
        st.write("Please upload an image to analyze.")
        process_image = False
else:
    # Use Webcam
    st.write("Please allow access to your webcam. Click 'Start' to begin.")
    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.frame = None

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame = img
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor, media_stream_constraints={
        "video": True,
        "audio": False,
    })

    process_image = False

    if ctx.video_processor:
        if st.button("Capture Image"):
            image = ctx.video_processor.frame
            if image is not None:
                process_image = True
            else:
                st.warning("No frame captured yet.")
    else:
        st.warning("Webcam is not ready yet. Please wait.")

if process_image:
    annotated_image = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Compute midpoints where required
            midpoint_13_14 = calculate_midpoint(landmarks[13], landmarks[14])

            # Updated ratio calculations
            ratio_1 = calculate_distance_from_midpoint(landmarks[8], midpoint_13_14) / calculate_distance(landmarks[2], landmarks[152])
            ratio_2 = calculate_distance(landmarks[8], landmarks[2]) / calculate_distance(landmarks[2], landmarks[152])
            ratio_3 = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance(landmarks[2], landmarks[152])
            ratio_4 = calculate_distance_from_midpoint(midpoint_13_14, landmarks[152]) / calculate_distance(landmarks[2], landmarks[152])
            ratio_5 = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance_from_midpoint(midpoint_13_14, landmarks[152])
            ratio_6 = calculate_distance_from_midpoint(landmarks[0], midpoint_13_14) / calculate_distance_from_midpoint(landmarks[2], midpoint_13_14)
            ratio_7 = calculate_distance_from_midpoint(landmarks[0], midpoint_13_14) / calculate_distance_from_midpoint(midpoint_13_14, landmarks[17])
            ratio_8 = calculate_distance(landmarks[219], landmarks[439]) / calculate_distance(landmarks[57], landmarks[287])
            ratio_9 = calculate_distance(landmarks[57], landmarks[287]) / calculate_distance(landmarks[130], landmarks[263])
            ratio_10 = calculate_distance(landmarks[219], landmarks[439]) / calculate_distance(landmarks[8], landmarks[2])
            ratio_11 = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance(landmarks[57], landmarks[287])
            ratio_12 = calculate_distance(landmarks[2], landmarks[152]) / calculate_distance(landmarks[57], landmarks[287])
            ratio_13 = calculate_distance(landmarks[8], landmarks[152]) / calculate_distance(landmarks[234], landmarks[454])
            ratio_14 = calculate_distance(landmarks[2], landmarks[0]) / calculate_distance_from_midpoint(landmarks[2], midpoint_13_14)
            ratio_15 = calculate_distance_from_midpoint(landmarks[0], midpoint_13_14) / calculate_distance(landmarks[2], landmarks[0])
            ratio_16 = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance(landmarks[2], landmarks[0])
            ratio_17 = calculate_distance_from_midpoint(landmarks[8], midpoint_13_14) / calculate_distance(landmarks[130], landmarks[263])
            ratio_18 = calculate_distance(landmarks[155], landmarks[463]) / calculate_distance(landmarks[219], landmarks[439])
            ratio_19 = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance_from_midpoint(landmarks[8], midpoint_13_14)
            ratio_20 = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance(landmarks[8], landmarks[2])
            ratio_21 = calculate_distance(landmarks[155], landmarks[463]) / calculate_distance(landmarks[130], landmarks[263])
            ratio_22 = calculate_distance(landmarks[463], landmarks[263]) / calculate_distance(landmarks[155], landmarks[463])

            # Draw lines for each ratio
            draw_line(annotated_image, landmarks[8], midpoint_13_14, color=(255, 0, 0))  # Ratio 1
            draw_line(annotated_image, landmarks[8], landmarks[2], color=(0, 255, 0))     # Ratio 2
            draw_line(annotated_image, landmarks[2], midpoint_13_14, color=(0, 0, 255))  # Ratio 3
            draw_line(annotated_image, midpoint_13_14, landmarks[152], color=(255, 255, 0)) # Ratio 4
            # For Ratio 5, lines are already drawn in Ratios 3 and 4
            draw_line(annotated_image, landmarks[0], midpoint_13_14, color=(255, 0, 255))  # Ratio 6
            draw_line(annotated_image, landmarks[0], landmarks[17], color=(0, 255, 255))   # Ratio 7
            draw_line(annotated_image, landmarks[219], landmarks[439], color=(128, 0, 128))# Ratio 8
            draw_line(annotated_image, landmarks[57], landmarks[287], color=(128, 128, 0)) # Ratio 9
            draw_line(annotated_image, landmarks[219], landmarks[2], color=(0, 128, 128))  # Ratio 10
            draw_line(annotated_image, landmarks[2], landmarks[57], color=(128, 0, 0))   # Ratio 11
            draw_line(annotated_image, landmarks[2], landmarks[152], color=(0, 128, 0))  # Ratio 12
            draw_line(annotated_image, landmarks[8], landmarks[152], color=(0, 0, 128))  # Ratio 13
            draw_line(annotated_image, landmarks[2], landmarks[0], color=(64, 64, 64))    # Ratio 14
            draw_line(annotated_image, landmarks[0], midpoint_13_14, color=(192, 192, 192))    # Ratio 15
            draw_line(annotated_image, landmarks[2], midpoint_13_14, color=(255, 165, 0))  # Ratio 16
            draw_line(annotated_image, landmarks[8], landmarks[130], color=(0, 0, 0))  # Ratio 17
            draw_line(annotated_image, landmarks[155], landmarks[463], color=(255, 105, 180))# Ratio 18
            draw_line(annotated_image, landmarks[2], landmarks[8], color=(75, 0, 130))    # Ratio 19
            # Ratio 20 uses the same line as Ratio 3
            draw_line(annotated_image, landmarks[155], landmarks[130], color=(0, 191, 255))# Ratio 21
            draw_line(annotated_image, landmarks[463], landmarks[263], color=(34, 139, 34))# Ratio 22

            # For vertical fifths

            # Calculate the total width from 127 to 447
            total_width = calculate_distance(landmarks[127], landmarks[447])

            # Calculate percentages
            percentage_width_127_130 = calculate_percentage_width(landmarks, 127, 130, total_width)
            percentage_width_130_133 = calculate_percentage_width(landmarks, 130, 133, total_width)
            percentage_width_133_463 = calculate_percentage_width(landmarks, 133, 463, total_width)
            percentage_width_463_263 = calculate_percentage_width(landmarks, 463, 263, total_width)
            percentage_width_263_447 = calculate_percentage_width(landmarks, 263, 447, total_width)

            # Draw vertical lines
            for point in [127, 130, 133, 463, 263, 447]:
                draw_vertical_line(annotated_image, landmarks[point], landmarks)

            # Data for the current image
            new_row_data = {
                'Ratio 1': ratio_1,
                'Ratio 2': ratio_2,
                'Ratio 3': ratio_3,
                'Ratio 4': ratio_4,
                'Ratio 5': ratio_5,
                'Ratio 6': ratio_6,
                'Ratio 7': ratio_7,
                'Ratio 8': ratio_8,
                'Ratio 9': ratio_9,
                'Ratio 10': ratio_10,
                'Ratio 11': ratio_11,
                'Ratio 12': ratio_12,
                'Ratio 13': ratio_13,
                'Ratio 14': ratio_14,
                'Ratio 15': ratio_15,
                'Ratio 16': ratio_16,
                'Ratio 17': ratio_17,
                'Ratio 18': ratio_18,
                'Ratio 19': ratio_19,
                'Ratio 20': ratio_20,
                'Ratio 21': ratio_21,
                'Ratio 22': ratio_22,
                'Percentage Width 127-130': percentage_width_127_130,
                'Percentage Width 130-133': percentage_width_130_133,
                'Percentage Width 133-463': percentage_width_133_463,
                'Percentage Width 463-263': percentage_width_463_263,
                'Percentage Width 263-447': percentage_width_263_447,
            }

            # Create a DataFrame from the new data
            results_df = pd.DataFrame([new_row_data])

            # Convert the annotated image to RGB for display
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # Display the results using columns
            st.markdown("---")
            st.subheader("Results")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(annotated_image_rgb, caption='Annotated Image', use_column_width=True)

            with col2:
                st.write("### Calculated Ratios")
                # Transpose the DataFrame to display ratios as rows
                results_df_transposed = results_df.T
                results_df_transposed.columns = ['Value']
                st.dataframe(results_df_transposed.style.format("{:.4f}"), height=600)

    else:
        st.error("No face detected in the image. Please try again.")

    face_mesh.close()
