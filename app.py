import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

# Initialize the webcam capture
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

# Initialize the face mesh detector with a maximum of 1 face
detector = FaceMeshDetector(maxFaces=1)

# Initialize the LivePlot for plotting blink ratio
plotY = LivePlot(640, 360, [20, 50], invert=True)

# List of landmarks for the eye region
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]

# Variables for blink detection
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect face mesh
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5, color, cv2.FILLED)

        # Get coordinates for vertical and horizontal eye distances
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        # Calculate distances
        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)

        # Draw lines for visualization
        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        # Calculate blink ratio
        ratio = int((lenghtVer / lenghtHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        # Blink detection logic
        if ratioAvg < 35 and counter == 0:
            blinkCounter += 1
            color = (0, 200, 0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (255, 0, 255)

        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), colorR=color)

        # Update the plot with the blink ratio
        imgPlot = plotY.update(ratioAvg, color)
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    # Show the stacked images
    cv2.imshow("Image", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()