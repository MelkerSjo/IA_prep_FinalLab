import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Unable to open camera")
    exit()

while True:
    # Capture frame by frame. 
    ret, frame = cap.read()

    # Check frame read correctly
    if not ret:
        print("Frame not recieved. Exiting...")
        break

    # Operations on the frame
    # Grayscale:
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    
    # Display the frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

# Everything done. release capture
cap.release()
cv.destroyAllWindows()