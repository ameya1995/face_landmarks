import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point2.y]) - np.array([point2.x, point2.y]))

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

def calculate_percentage_width(landmarks, point1, point2, total_width):
    """Calculate the width between two points as a percentage of total width."""
    width = calculate_horizontal_distance(landmarks[point1], landmarks[point2])
    return (width / total_width) * 100

# Helper function: draw a circle (point) on the image
def draw_point(image, point_or_landmark, color=(0, 0, 255), radius=10, text=None):
    """
    Draws a circle at the given point or landmark.
    `point_or_landmark` can be a dict (midpoint) or a Mediapipe landmark.
    """
    height, width, _ = image.shape

    if isinstance(point_or_landmark, dict):
        x = int(point_or_landmark['x'] * width)
        y = int(point_or_landmark['y'] * height)
    else:  # Mediapipe landmark
        x = int(point_or_landmark.x * width)
        y = int(point_or_landmark.y * height)

    # Red circle
    cv2.circle(image, (x, y), radius, (0, 0, 255), -1)

    # Blue text
    if text is not None:
        cv2.putText(
            image,
            text,
            (x + 5, y - 5),  # offset so text doesn't overlap the circle
            cv2.FONT_HERSHEY_SIMPLEX,
            2,    # new, larger font scale
            (255, 0, 0),
            4       # optional increase in thickness
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

                # Ratio calculations
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

                # Draw key points & midpoints
                key_points = [
                    (2, "Sn"), (8, "N'"), (13, "St"), (17, "Li"),
                    (152, "Me'"), (219, "Alr"), (439, "All"),
                    (130, "Exr"), (263, "Exl"), (234, "Zyr"), (454, "Zyl"),
                    (155, "Enr"), (463, "Enl"), (291, "Chl")
                ]
                for (idx, label) in key_points:
                    draw_point(annotated_image, landmarks[idx], color=(255, 0, 0), radius=10, text=label)

                # Midpoints
                # draw_point(annotated_image, midpoint_13_14, color=(0, 255, 255), radius=10, text="mid_13_14")
                draw_point(annotated_image, midpoint_57_61, color=(255, 255, 0), radius=10, text="Chr")
                # draw_point(annotated_image, midpoint_291_287, color=(255, 255, 0), radius=10, text="mid_291_287")

                # For vertical fifths (width percentages)
                total_width = calculate_distance(landmarks[127], landmarks[447])

                percentage_width_130_133 = calculate_percentage_width(landmarks, 130, 133, total_width)
                percentage_width_133_463 = calculate_percentage_width(landmarks, 133, 463, total_width)
                percentage_width_463_263 = calculate_percentage_width(landmarks, 463, 263, total_width)

                # Annotate those specific points too
                # draw_point(annotated_image, landmarks[127], color=(0, 255, 0), radius=10, text="127")
                # draw_point(annotated_image, landmarks[130], color=(0, 255, 0), radius=10, text="130")
                # draw_point(annotated_image, landmarks[133], color=(0, 255, 0), radius=10, text="133")
                # draw_point(annotated_image, landmarks[463], color=(0, 255, 0), radius=10, text="463")
                # draw_point(annotated_image, landmarks[263], color=(0, 255, 0), radius=10, text="263")
                # draw_point(annotated_image, landmarks[447], color=(0, 255, 0), radius=10, text="447")

                # Create a copy of the annotated image for width-proportion lines
                annotated_image_width = annotated_image.copy()

                # Retrieve image dimensions for coordinate calculation
                height, width, _ = annotated_image_width.shape

                def get_xy(landmark):
                    return (
                        int(landmark.x * width),
                        int(landmark.y * height)
                    )

                # Coordinates for each relevant landmark
                pt_130 = get_xy(landmarks[130])
                pt_133 = get_xy(landmarks[133])
                pt_463 = get_xy(landmarks[463])
                pt_263 = get_xy(landmarks[263])

                # Draw lines in yellow
                cv2.line(annotated_image_width, pt_130, pt_133, (0, 255, 255), 2)
                cv2.line(annotated_image_width, pt_133, pt_463, (0, 255, 255), 2)
                cv2.line(annotated_image_width, pt_463, pt_263, (0, 255, 255), 2)

                # Helper to place text roughly at the midpoint of two points
                def draw_line_text(image, p1, p2, label):
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2
                    cv2.putText(
                        image,
                        label,
                        (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 255),
                        3
                    )

                draw_line_text(annotated_image_width, pt_130, pt_133, f"A: {percentage_width_130_133:.2f}%")
                draw_line_text(annotated_image_width, pt_133, pt_463, f"B: {percentage_width_133_463:.2f}%")
                draw_line_text(annotated_image_width, pt_463, pt_263, f"C: {percentage_width_463_263:.2f}%")

                # Save the new image
                annotated_image_width_path = os.path.join('Output_images', "annotated_width_" + file_name)
                cv2.imwrite(annotated_image_width_path, annotated_image_width)

                # Create a fresh copy for vertical width lines without other point annotations
                annotated_image_width = image.copy()

                height, width, _ = annotated_image_width.shape

                # Get x-coordinates of landmarks
                x_130 = int(landmarks[130].x * width)
                x_133 = int(landmarks[133].x * width)
                x_463 = int(landmarks[463].x * width)
                x_263 = int(landmarks[263].x * width)

                # Draw vertical lines in yellow
                cv2.line(annotated_image_width, (x_130, 0), (x_130, height), (0, 255, 255), 2)
                cv2.line(annotated_image_width, (x_133, 0), (x_133, height), (0, 255, 255), 2)
                cv2.line(annotated_image_width, (x_463, 0), (x_463, height), (0, 255, 255), 2)
                cv2.line(annotated_image_width, (x_263, 0), (x_263, height), (0, 255, 255), 2)

                # Helper function for labeling distance between two x-coordinates
                def draw_vertical_distance_text(image, x1, x2, label):
                    mid_x = (x1 + x2) // 2
                    # Place text near the top
                    cv2.putText(
                        image,
                        label,
                        (mid_x, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 255),
                        3
                    )

                # Use the previously computed percentage widths to place labels
                draw_vertical_distance_text(annotated_image_width, x_130, x_133, f"A: {percentage_width_130_133:.2f}%")
                draw_vertical_distance_text(annotated_image_width, x_133, x_463, f"B: {percentage_width_133_463:.2f}%")
                draw_vertical_distance_text(annotated_image_width, x_463, x_263, f"C: {percentage_width_463_263:.2f}%")

                # Write out the new image
                annotated_image_width_path = os.path.join('Output_images', "annotated_width_" + file_name)
                cv2.imwrite(annotated_image_width_path, annotated_image_width)

                # Create a fresh copy for vertical width lines without other point annotations
                annotated_image_width = image.copy()

                height, width, _ = annotated_image_width.shape

                # Get x-coordinates of landmarks
                x_130 = int(landmarks[130].x * width)
                x_133 = int(landmarks[133].x * width)
                x_463 = int(landmarks[463].x * width)
                x_263 = int(landmarks[263].x * width)

                # Draw vertical lines in yellow
                cv2.line(annotated_image_width, (x_130, 0), (x_130, height), (0, 255, 255), 2)
                cv2.line(annotated_image_width, (x_133, 0), (x_133, height), (0, 255, 255), 2)
                cv2.line(annotated_image_width, (x_463, 0), (x_463, height), (0, 255, 255), 2)
                cv2.line(annotated_image_width, (x_263, 0), (x_263, height), (0, 255, 255), 2)

                # Helper function to place "1/3" near the top and "Proportion X" in the middle
                def draw_two_label_text(image, x1, x2, fraction_text, proportion_text):
                    mid_x = ((x1 + x2) // 2) - 50

                    # Move "1/3" slightly down
                    fraction_text_y = 1700
                    cv2.putText(
                        image,
                        fraction_text,
                        (mid_x, fraction_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.6,
                        (255, 0, 0),
                        6
                    )

                    # Shift "Proportion X" left and place in middle
                    mid_y = (height // 2) + 200
                    proportion_offset = 150
                    cv2.putText(
                        image,
                        proportion_text,
                        (mid_x - proportion_offset, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.6,
                        (255, 0, 0),
                        6
                    )

                # Label each segment
                draw_two_label_text(annotated_image_width, x_130, x_133, "1/3", "Proportion A")
                draw_two_label_text(annotated_image_width, x_133, x_463, "1/3", "Proportion B")
                draw_two_label_text(annotated_image_width, x_463, x_263, "1/3", "Proportion C")

                annotated_image_width_path = os.path.join('Output_images', "annotated_width_" + file_name)
                cv2.imwrite(annotated_image_width_path, annotated_image_width)

                # Create a copy of the source image for drawing lines
                annotate_lines_image = image.copy()

                # Blue ratio lines with thickness=6
                draw_line(annotate_lines_image, landmarks[8], midpoint_13_14, (255, 0, 0), 6)
                draw_line(annotate_lines_image, landmarks[8], landmarks[2], (255, 0, 0), 6)
                draw_line(annotate_lines_image, landmarks[2], midpoint_13_14, (255, 0, 0), 6)
                draw_line(annotate_lines_image, midpoint_13_14, landmarks[152], (255, 0, 0), 6)
                # ...turn all other ratio lines blue with thickness=6...
                draw_line(annotate_lines_image, landmarks[0], midpoint_13_14, (255, 0, 0), 6)   # ratio_6
                draw_line(annotate_lines_image, midpoint_13_14, landmarks[17], (255, 0, 0), 6)  # ratio_7
                draw_line(annotate_lines_image, landmarks[219], landmarks[439], (255, 0, 0), 6) # ratio_8
                draw_line(annotate_lines_image, landmarks[57], landmarks[287], (255, 0, 0), 6)  # ratio_9
                draw_line(annotate_lines_image, landmarks[219], landmarks[2], (255, 0, 0), 6)   # ratio_10
                draw_line(annotate_lines_image, landmarks[2], landmarks[57], (255, 0, 0), 6)    # ratio_11
                draw_line(annotate_lines_image, landmarks[2], landmarks[152], (255, 0, 0), 6)   # ratio_12
                draw_line(annotate_lines_image, landmarks[8], landmarks[152], (255, 0, 0), 6)   # ratio_13
                draw_line(annotate_lines_image, landmarks[2], landmarks[0], (255, 0, 0), 6)     # ratio_14
                draw_line(annotate_lines_image, landmarks[0], midpoint_13_14, (255, 0, 0), 6)   # ratio_15
                draw_line(annotate_lines_image, landmarks[2], midpoint_13_14, (255, 0, 0), 6)   # ratio_16
                draw_line(annotate_lines_image, landmarks[8], landmarks[130], (255, 0, 0), 6)   # ratio_17
                draw_line(annotate_lines_image, landmarks[155], landmarks[463], (255, 0, 0), 6) # ratio_18
                draw_line(annotate_lines_image, landmarks[2], landmarks[8], (255, 0, 0), 6)     # ratio_19
                # (ratio_20 overlaps existing lines)
                draw_line(annotate_lines_image, landmarks[155], landmarks[130], (255, 0, 0), 6) # ratio_21
                draw_line(annotate_lines_image, landmarks[463], landmarks[263], (255, 0, 0), 6) # ratio_22

                # Also add the vertical lines in yellow
                height, width, _ = annotate_lines_image.shape
                for pt_index in [130, 133, 463, 263]:
                    x_pt = int(landmarks[pt_index].x * width)
                    cv2.line(annotate_lines_image, (x_pt, 0), (x_pt, height), (0, 255, 255), 2)

                annotate_lines_path = os.path.join('Output_images', "annotate_lines_" + file_name)
                cv2.imwrite(annotate_lines_path, annotate_lines_image)

                # Store data in DataFrame
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
                    'Percentage Width 130-133': percentage_width_130_133,
                    'Percentage Width 133-463': percentage_width_133_463,
                    'Percentage Width 463-263': percentage_width_463_263,
                }
                new_row_df = pd.DataFrame([new_row_data])
                results_df = pd.concat([results_df, new_row_df], ignore_index=True)
            
        # Save the annotated image
        annotated_image_path = os.path.join('Output_images', "annotated_" + file_name)
        cv2.imwrite(annotated_image_path, annotated_image)

# Close MediaPipe Face Mesh
face_mesh.close()

# Display or save the results
print(results_df)
# Optionally, save to a CSV file
results_df.to_csv('output.csv', index=False)
