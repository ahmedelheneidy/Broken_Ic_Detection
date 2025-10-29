import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture("Video_20221206114214150.avi")

while True:
    # Capture a frame from the video
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to the frame
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Find the contours in the frame
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the count and missing legs variables
    count = 0
    missing_legs = []

    # Iterate through the contours
    for contour in contours:
        # Check if the contour is a rectangle
        if len(contour) == 4:
            # Increment the count
            count += 1
        else:
            # Add the contour to the list of missing legs
            missing_legs.append(contour)

    # Draw the contours on the frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Display the count and missing legs on the frame
    cv2.putText(frame, "Number of legs: " + str(count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Missing legs: " + str(missing_legs), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("IC legs detection", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Close all the windows
cv2.destroyAllWindows()
