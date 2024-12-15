import cv2
import utlis

webcam = False
path = 'ppq.jpeg'  # Path to your image
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)

# Target width and height for the A4 sheet
wP = 210 * 3  # A4 width in mm (scaled for visualization)
hP = 297 * 3  # A4 height in mm (scaled for visualization)

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    # Find contours in the image
    img, finalContours = utlis.getContours(img, showCanny=False, minArea=5000, filter=4)

    # Ensure at least one contour is detected
    if len(finalContours) != 0:
        biggest = finalContours[0][2]  # Biggest rectangle contour
        aspect_ratio = wP / hP

        # Warp the detected A4 sheet to the target dimensions
        imgWarp = utlis.warpImg(img, biggest, wP, hP)

        # Detect smaller contours inside the warped A4 sheet
        img2, finalContours2 = utlis.getContours(
            imgWarp, showCanny=False, minArea=2000, filter=4, cThr=[50, 50], draw=True
        )

        # Ensure contours inside the A4 sheet are detected
        if len(finalContours2) != 0:
            for obj in finalContours2:
                # Draw contours on the warped image
                cv2.polylines(img2, [obj[2]], True, (0, 255, 0), thickness=2)
                nPoints = utlis.reorder(obj[2])

                # Calculate width and height (in cm, assuming 10px = 1mm scaling)
                mW = round((utlis.findDis(nPoints[0][0] // 3, nPoints[1][0] // 3) / 10), 1)
                mH = round((utlis.findDis(nPoints[0][0] // 3, nPoints[2][0] // 3) / 10), 1)

                # Draw arrowed lines for width and height
                cv2.arrowedLine(
                    img2,
                    (nPoints[0][0][0], nPoints[0][0][1]),
                    (nPoints[1][0][0], nPoints[1][0][1]),
                    (255, 0, 0),
                    3,
                    tipLength=0.05,
                )
                cv2.arrowedLine(
                    img2,
                    (nPoints[0][0][0], nPoints[0][0][1]),
                    (nPoints[2][0][0], nPoints[2][0][1]),
                    (255, 0, 0),
                    3,
                    tipLength=0.05,
                )

                # Display dimensions as text
                cv2.putText(
                    img2,
                    f'Width: {mW} cm',
                    (nPoints[0][0][0] + 20, nPoints[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    img2,
                    f'Height: {mH} cm',
                    (nPoints[0][0][0] - 100, nPoints[0][0][1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

        # Show the warped image with dimensions
        cv2.imshow('Warped Image (A4 Sheet)', img2)

    # Display the original frame (resized)
    img_resized = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow('Original Image', img_resized)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
