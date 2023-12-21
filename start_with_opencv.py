import cv2
import cv2.data
import os
import random


image = cv2.imread('./assets/lenna.png')
cv2.imshow("Real image", image)



image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", image_gray)



image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image as rgb.
# may show more blue image becouse of cv2 reads image colors as BGR (reverse of RGB)
cv2.imshow("RGB image", image_rgb)


image_cut = image[0:100,0:100]
# cut image 100x100 px from 0x0 points, first Y cords then X cords given
cv2.imshow("Image cut", image_cut)


join_images_vertical = cv2.vconcat([image, image_rgb])
cv2.imshow("Join images (vertical)", join_images_vertical)

join_images_horizontal = cv2.hconcat([image, image_rgb])
cv2.imshow("Join images (horizontal)", join_images_horizontal)


_, thresh = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY) 
cv2.imshow("Threshold", thresh)


grayscale_plus_real = cv2.add(cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR), image)
cv2.imshow("Gray + Real image", grayscale_plus_real)


shapes = cv2.imread('./assets/shapes.png')
cv2.imshow("Shapes image", shapes)

shapes_roi = shapes[10:shapes.shape[0] - 10, 10:shapes.shape[1] - 10]

blur_image = cv2.GaussianBlur(shapes_roi, (5, 5), 1)
cv2.imshow("Blur image", blur_image)

canny_image = cv2.Canny(blur_image, 0, 128)
cv2.imshow("Canny image", canny_image)

contours, _ = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cntrRect = []
for i in contours:
    epsilon = 0.05 * cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, epsilon, True)
    if len(approx) == 4:
        cntrRect.append(approx)

cv2.drawContours(shapes_roi, cntrRect, -1, (0, 0, 255), 5)
cv2.imshow('Shapes detected', shapes_roi)


face_cascader = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt.xml'))
cam = cv2.VideoCapture(random.choice(['./assets/nice.mp4', 0]))

while cam.isOpened():
    _, cap = cam.read()
    if not _:
        break
    
    for x, y, w, h in face_cascader.detectMultiScale(cap):
        cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 4)

    cap = cv2.resize(cap, (800, 600))
    cv2.imshow("Video", cap)
    
    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()
