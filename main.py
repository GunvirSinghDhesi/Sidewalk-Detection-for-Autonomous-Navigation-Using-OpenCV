import cv2
import numpy as np

def apply_gabor_filter(image):
    gabor_kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_kernels.append(kernel)
    
    filtered_images = [cv2.filter2D(image, cv2.CV_8UC3, kernel) for kernel in gabor_kernels]
    combined = np.sum(filtered_images, axis=0)
    return combined

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255  # White color mask
    cv2.fillPoly(mask, np.array([vertices], np.int32), match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def find_footpath_midpoint(lines, frame_center):
    left_x_positions = []
    right_x_positions = []
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 < frame_center and x2 < frame_center:
            left_lines.append((x1, y1))
            left_lines.append((x2, y2))
            left_x_positions.extend([x1, x2])
        elif x1 > frame_center and x2 > frame_center:
            right_lines.append((x1, y1))
            right_lines.append((x2, y2))
            right_x_positions.extend([x1, x2])
    
    if not left_x_positions or not right_x_positions:
        return None

    # Calculate the average x position for left and right edges
    left_edge = np.mean(left_x_positions)
    right_edge = np.mean(right_x_positions)

    # Calculate the midpoint between the left and right edges (footpath center)
    footpath_center_x = (left_edge + right_edge) / 2

    # Find the average y position for better midpoint placement
    average_y = np.mean([y for _, y in left_lines + right_lines])

    return int(left_edge), int(right_edge), int(footpath_center_x), int(average_y)

# Use the video path provided
video = cv2.VideoCapture("./test1.mp4")

while True:
    ret, or_frame = video.read()
    if not ret:
        break
    
    frame = cv2.GaussianBlur(or_frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_gray = np.array([0, 0, 120])
    upper_gray = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    edges = cv2.Canny(mask, 50, 150)
    gabor_result = apply_gabor_filter(frame)
    
    # Normalize and resize the Gabor filter result to match the edges size
    gabor_result_uint8 = cv2.normalize(gabor_result, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    gabor_result_resized = cv2.resize(gabor_result_uint8, (edges.shape[1], edges.shape[0]))

    # Convert the Gabor result to grayscale to match the edges' single channel
    gabor_result_gray = cv2.cvtColor(gabor_result_resized, cv2.COLOR_BGR2GRAY)

    # Perform the bitwise operation
    combined_mask = cv2.bitwise_and(edges, gabor_result_gray)
    
    roi_vertices = [(0, frame.shape[0]), (frame.shape[1], frame.shape[0]), (frame.shape[1], int(frame.shape[0]*0.5)), (0, int(frame.shape[0]*0.5))]
    roi_combined = region_of_interest(combined_mask, roi_vertices)
    
    lines = cv2.HoughLinesP(roi_combined, 1, np.pi/180, 30, minLineLength=30, maxLineGap=100)
    
    if lines is not None:
        # Draw all the detected lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines for all detected edges

        # Calculate the center of the frame
        frame_center = frame.shape[1] // 2

        # Find the edges and midpoint of the footpath
        footpath_info = find_footpath_midpoint(lines, frame_center)

        if footpath_info:
            left_edge, right_edge, footpath_center_x, average_y = footpath_info
            height = frame.shape[0]
            width = frame.shape[1]

            # Draw the three lines for distance measurement
            cv2.line(frame, (frame_center, height), (left_edge, average_y), (0, 255, 0), 2)  # Left distance line
            cv2.line(frame, (frame_center, height), (right_edge, average_y), (0, 255, 0), 2)  # Right distance line
            cv2.line(frame, (frame_center, height), (footpath_center_x, average_y), (0, 255, 0), 2)  # Forward distance line

            # Optionally, you can annotate the distances on the frame
            left_distance = np.sqrt((left_edge - 0) ** 2 + (average_y - height) ** 2)
            right_distance = np.sqrt((right_edge - width) ** 2 + (average_y - height) ** 2)
            forward_distance = np.sqrt((footpath_center_x - frame_center) ** 2 + (average_y - height) ** 2)

            cv2.putText(frame, f"Left Distance: {left_distance:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Distance: {right_distance:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Forward Distance: {forward_distance:.2f}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("edges", roi_combined)
    key = cv2.waitKey(25)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
