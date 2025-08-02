import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

def calculate_horizontal_distance(point1, point2):
    """Calculate the horizontal distance between two points."""
    return abs(point1.x - point2.x)

# Function to calculate the midpoint between two points
def calculate_midpoint(point1, point2):
    return {'x': (point1.x + point2.x) / 2, 'y': (point1.y + point2.y) / 2}

def calculate_custom_30_x_left(point1, point2):
    return {'x': point1.x + ((point2.x - point1.x) / 4), 'y': (point1.y + point2.y) / 2}

def calculate_custom_30_x_right(point1, point2):
    return {'x': point1.x + (3 * (point2.x - point1.x) / 4), 'y': (point1.y + point2.y) / 2}

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

def draw_line(image, landmark1, landmark2, color=(255, 0, 0), thickness=10):
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

def draw_vertical_line(image, landmark, color=(0, 255, 200), thickness=10):
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

# Path to the folder containing images
folder_path = 'Sequence vise order'

# DataFrame to store results
results_df = pd.DataFrame()

# Process each image in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(('.png', '.JPG', '.jpeg')):
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)
        annotated_image = image.copy()  # Create a copy of the image for drawing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # Compute midpoints where required
                midpoint_13_14 = calculate_midpoint(landmarks[13], landmarks[14])
                midpoint_57_61 = calculate_custom_30_x_right(landmarks[57], landmarks[61])
                midpoint_291_287 = calculate_custom_30_x_left(landmarks[291], landmarks[287])

                # Updated ratio calculations
                ratio_1 = calculate_distance_from_midpoint(landmarks[8], midpoint_13_14) / calculate_distance(landmarks[2], landmarks[152])
                ratio_2 = calculate_distance(landmarks[8], landmarks[2]) / calculate_distance(landmarks[2], landmarks[152])
                ratio_3 = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance(landmarks[2], landmarks[152])
                ratio_4 = calculate_distance_from_midpoint(midpoint_13_14, landmarks[152]) / calculate_distance(landmarks[2], landmarks[152])
                ratio_5 = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance_from_midpoint(midpoint_13_14, landmarks[152])
                ratio_6 = calculate_distance_from_midpoint(landmarks[0], midpoint_13_14) / calculate_distance_from_midpoint(landmarks[2], midpoint_13_14)
                ratio_7 = calculate_distance_from_midpoint(landmarks[0], midpoint_13_14) / calculate_distance_from_midpoint(midpoint_13_14, landmarks[17])
                ratio_8 = calculate_distance(landmarks[219], landmarks[439]) / calculate_distance_from_midpoint(midpoint_57_61, landmarks[291])
                ratio_9 = calculate_distance_from_midpoint(midpoint_57_61, landmarks[291]) / calculate_distance(landmarks[130], landmarks[263])
                ratio_10 = calculate_distance(landmarks[219], landmarks[439]) / calculate_distance(landmarks[8], landmarks[2])
                ratio_11 = calculate_distance_from_midpoint(landmarks[2], midpoint_13_14) / calculate_distance_from_midpoint(midpoint_57_61, landmarks[291])
                ratio_12 = calculate_distance(landmarks[2], landmarks[152]) / calculate_distance_from_midpoint(midpoint_57_61, landmarks[291])
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
                draw_line(annotated_image, landmarks[8], midpoint_13_14)  # Ratio 1
                draw_line(annotated_image, landmarks[8], landmarks[2])     # Ratio 2
                draw_line(annotated_image, landmarks[2], midpoint_13_14)  # Ratio 3
                draw_line(annotated_image, midpoint_13_14, landmarks[152]) # Ratio 4
                # For Ratio 5, lines are already drawn in Ratios 3 and 4
                draw_line(annotated_image, landmarks[0], midpoint_13_14)  # Ratio 6
                draw_line(annotated_image, landmarks[0], landmarks[17])   # Ratio 7
                draw_line(annotated_image, landmarks[219], landmarks[439])# Ratio 8
                draw_line(annotated_image, midpoint_57_61, landmarks[291]) # Ratio 9
                draw_line(annotated_image, landmarks[219], landmarks[2])  # Ratio 10
                draw_line(annotated_image, landmarks[2], midpoint_57_61)   # Ratio 11
                draw_line(annotated_image, landmarks[2], landmarks[152])  # Ratio 12
                draw_line(annotated_image, landmarks[8], landmarks[152])  # Ratio 13
                draw_line(annotated_image, landmarks[2], landmarks[0])    # Ratio 14
                draw_line(annotated_image, landmarks[0], landmarks[0])    # Ratio 15
                draw_line(annotated_image, landmarks[2], midpoint_13_14)  # Ratio 16
                draw_line(annotated_image, landmarks[8], landmarks[130])  # Ratio 17
                draw_line(annotated_image, landmarks[155], landmarks[463])# Ratio 18
                draw_line(annotated_image, landmarks[2], landmarks[8])    # Ratio 19
                draw_line(annotated_image, landmarks[2], midpoint_13_14)  # Ratio 20
                draw_line(annotated_image, landmarks[155], landmarks[130])# Ratio 21
                draw_line(annotated_image, landmarks[463], landmarks[263])# Ratio 22
                
                # For vertical fifths
                
                # Calculate the total width from 127 to 447
                total_width = calculate_distance(landmarks[127], landmarks[447])

                # Calculate percentages
                percentage_width_127_130 = calculate_percentage_width(landmarks, 127, 130, total_width)
                percentage_width_130_133 = calculate_percentage_width(landmarks, 130, 133, total_width)
                percentage_width_133_463 = calculate_percentage_width(landmarks, 133, 463, total_width)
                percentage_width_463_263 = calculate_percentage_width(landmarks, 463, 263, total_width)
                percentage_width_263_447 = calculate_percentage_width(landmarks, 263, 447, total_width)
                
                # Draw vertical lines 359
                for point in [127, 130, 133, 463, 263, 447]:
                    draw_vertical_line(annotated_image, landmarks[point])


                # Data for the current image
                new_row_data = {
                    'Image': file_name,
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
                new_row_df = pd.DataFrame([new_row_data])

                # Concatenate the new DataFrame with the existing one
                results_df = pd.concat([results_df, new_row_df], ignore_index=True)
            
        annotated_image_path = os.path.join('Output_images', "annotated_" + file_name)
        cv2.imwrite(annotated_image_path, annotated_image)


# Close MediaPipe Face Mesh
face_mesh.close()

# Display or save the results
print(results_df)
# Optionally, save to a CSV file
results_df.to_csv('output.csv', index=False)
