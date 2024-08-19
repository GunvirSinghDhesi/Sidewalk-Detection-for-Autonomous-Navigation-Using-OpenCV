import cv2
import numpy as np
import os
import math

# Load video
videoFrame = cv2.VideoCapture(os.path.join(os.getcwd(), "test1.mp4"))





# Reference Image pixels/cm conversion ------------------------------------------------------------------------------------------------------------------------------  
# Load the reference image
reference_image = cv2.imread('reference_image.jpg')

# Manually define the known points (in pixels) and the real-world distance between them (in cm)
point1 = (0, 0)  # Coordinates in pixels
point2 = (0, 0)  # Coordinates in pixels
known_distance_cm = 100.0  # The real-world distance between point1 and point2 in cm

# Calculate the pixel distance
pixel_distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Calculate the pixel-to-centimeter ratio
pixels_per_cm = pixel_distance / known_distance_cm











# Edge Distance Lines ------------------------------------------------------------------------------------------------------------------------------
def calculate_intersection(contours, angle, bottom_center, max_distance):
    angle_rad = np.deg2rad(angle)
    x_dir = np.cos(angle_rad)
    y_dir = np.sin(angle_rad)

    for d in range(1, max_distance):
        x = int(bottom_center[0] + d * x_dir)
        y = int(bottom_center[1] - d * y_dir)

        if x < 0 or x >= width or y < 0 or y >= height:
            break

        for contour in contours:
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                return (x, y), d
    
    return (int(bottom_center[0] + max_distance * x_dir), int(bottom_center[1] - max_distance * y_dir)), max_distance

# Edge Distance Lines
def distanceContours(or_frame, edges, bottom_center, angles):
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Maximum possible distance in the frame (diagonal)
    max_distance = int(np.sqrt(or_frame.shape[0]**2 + or_frame.shape[1]**2))

    for angle in angles:
        end_point, distance = calculate_intersection(contours, angle, bottom_center, max_distance)
        color = (0, 255, 0) if angle == 90 else (0, 0, 255)
        cv2.line(or_frame, bottom_center, end_point, color, 2)
        cv2.putText(or_frame, f'{angle} Deg Dist: {distance:.2f}', (10, 30 + 30 * angles.index(angle)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return or_frame




def centerSideWalk():
    pass    





while True:
    ret, or_frame = videoFrame.read()
    if not ret:
        videoFrame = cv2.VideoCapture(os.path.join(os.getcwd(), "test1.mp4"))
        continue

    # BLUR MASK ------------------------------------------------------------------------------------------------------------------------------
    blurred = cv2.GaussianBlur(or_frame, (13, 13), 0)

    # HSV MASK ------------------------------------------------------------------------------------------------------------------------------
    # Convert to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Define the range of the color for the sidewalk
    lower_bound = np.array([0, 0, 120])
    upper_bound = np.array([180, 50, 255])
    
    
    
    
    # Morphological Operations ------------------------------------------------------------------------------------------------------------------------------

   # Create a mask to isolate the sidewalk
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply morphological operations to remove small objects
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small objects
    
    # Additional dilation to make sure the sidewalk is continuous
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    
    # COLOR FILTER ------------------------------------------------------------------------------------------------------------------------------

    # Apply the mask to the original frame
    colorFilter = cv2.bitwise_and(blurred, blurred, mask=mask)



    # ROI ------------------------------------------------------------------------------------------------------------------------------
    
    # Define a Region of Interest (ROI) for the sidewalk (e.g., bottom half of the image)
    height, width = or_frame.shape[:2]
    bottom_center = (width // 2, height - 1)

    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(roi_mask, (0, height//2), (width, height), (255), thickness=cv2.FILLED)
    
    # Combine the ROI mask with the color filter mask
    final_mask = cv2.bitwise_and(mask, roi_mask)

    # Edge Detection ------------------------------------------------------------------------------------------------------------------------------
    edges = cv2.Canny(final_mask, 100, 150)



    # Contour Lines ------------------------------------------------------------------------------------------------------------------------------
    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    # Filter contours by area to remove small objects
    min_area = 500  # Adjust this threshold based on your needs
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    
    # Contour Lines ------------------------------------------------------------------------------------------------------------------------------
    angles = [45, 90, 135]
    or_frame = distanceContours(or_frame, edges, bottom_center, angles)
    # Draw contours on the original frame
    cv2.drawContours(or_frame, contours, -1, (0, 255, 0), 3)
   




    # Show the result
    cv2.imshow('Original', or_frame)
    cv2.imshow('Color Filter', colorFilter)
    cv2.imshow('Edges', edges)



 

    # Exiting the player
    key = cv2.waitKey(25)
    if key == 27:
        videoFrame.release()
        break

cv2.destroyAllWindows()
