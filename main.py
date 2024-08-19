import cv2
import numpy as np
import os

# Load video
videoFrame = cv2.VideoCapture(os.path.join(os.getcwd(), "test1.mp4"))

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
