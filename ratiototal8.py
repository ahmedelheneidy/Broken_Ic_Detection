import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to the frame
    _, thresh = cv2.threshold(gray, 246, 255, cv2.THRESH_BINARY)

    # Find the contours in the frame
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the count and missing legs variables
    count = 0
    missing_legs = []

    # Iterate through the contours
    for contour in contours:
        # Check if the contour is a rectangle
        if len(contour) == 4:
            # Get the bounding rectangle of the contour
            rect = cv2.minAreaRect(contour)
            (x, y), (w, h), angle = rect
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 0.9 and aspect_ratio < 1.1:
                count += 1
            else:
                missing_legs.append(contour)
    # Draw the contours on the frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Display the count and missing legs on the frame
    cv2.putText(frame, "Number of legs: " + str(count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Missing legs: " + str(8-count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("IC legs detection", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Close all the windows
cv2.destroyAllWindows()
