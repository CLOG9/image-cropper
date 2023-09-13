# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread('./IMG_20230913_114451.jpg')

# # Convert the image to the HSV color space
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define the lower and upper bounds of the green color in HSV
# lower_green = np.array([35, 50, 50])
# upper_green = np.array([85, 255, 255])

# # Create a mask to isolate the green color
# mask = cv2.inRange(hsv, lower_green, upper_green)

# # Apply a Gaussian blur to the mask to reduce noise
# blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)

# # Find contours in the blurred mask
# contours, _ = cv2.findContours(
#     blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Create an empty mask to draw the filled contour of the ID card
# filled_mask = np.zeros_like(image)

# # Draw the largest contour (assuming it's the ID card) on the filled mask
# if contours:
#     largest_contour = max(contours, key=cv2.contourArea)
#     cv2.drawContours(filled_mask, [largest_contour],
#                      0, (255, 255, 255), thickness=cv2.FILLED)

# # Apply the filled mask to the original image to remove the background
# result = cv2.bitwise_and(image, filled_mask)

# # Save the result
# cv2.imwrite('id_card_removed_background.jpg', result)

# # Display the result
# cv2.imshow('ID Card with Background Removed', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread('./IMG_20230913_114451.jpg')

# # Convert the image to the HSV color space
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define the lower and upper bounds of the green color in HSV
# lower_green = np.array([30, 40, 40])
# upper_green = np.array([90, 255, 255])

# # Create a mask to isolate the green color
# mask = cv2.inRange(hsv, lower_green, upper_green)

# # Apply a stronger Gaussian blur to the mask to reduce noise
# blurred_mask = cv2.GaussianBlur(mask, (25, 25), 0)

# # Find contours in the blurred mask
# contours, _ = cv2.findContours(
#     blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Find the bounding box of the largest contour (assuming it's the ID card)
# if contours:
#     largest_contour = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(largest_contour)

#     # Crop the image to the bounding box
#     id_card_cropped = image[y:y+h, x:x+w]

#     # Save the cropped image
#     cv2.imwrite('id_card_cropped.jpg', id_card_cropped)

#     # Display the cropped image
#     cv2.imshow('Cropped ID Card', id_card_cropped)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("No ID card found in the image.")


import cv2
import numpy as np
import os

# Get the user's home directory
home_dir = os.path.expanduser("~")
# Construct the desktop path
desktop_path = os.path.join(home_dir, "Desktop")
imagePath = input("Add the image name here >> ")
# Load the image
image = cv2.imread(f"{desktop_path}\{imagePath}")

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds of the green color in HSV
lower_green = np.array([30, 40, 40])
upper_green = np.array([90, 255, 255])

# Create a mask to isolate the green color
mask = cv2.inRange(hsv, lower_green, upper_green)

# Apply a stronger Gaussian blur to the mask to reduce noise
blurred_mask = cv2.GaussianBlur(mask, (25, 25), 0)

# Find contours in the blurred mask
contours, _ = cv2.findContours(
    blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the bounding box of the largest contour (assuming it's the ID card)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the angle of rotation to make the ID card horizontal
    angle = cv2.minAreaRect(largest_contour)[-1]

    # Ensure the angle is in the range (-45, 45) degrees
    if angle < -45:
        angle += 90

    # Rotate the image by the calculated angle
    center = (x + w // 2, y + h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (image.shape[1], image.shape[0]))

    # Crop the rotated image to the bounding box
    rotated_id_card = rotated_image[y:y+h, x:x+w]

    # Calculate the orientation of card edges using Hough Line Transform
    gray_rotated = cv2.cvtColor(rotated_id_card, cv2.COLOR_BGR2GRAY)
    edges_rotated = cv2.Canny(gray_rotated, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges_rotated, 1, np.pi / 180, 100)

    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Calculate the angle of the line
            line_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Adjust the rotation to align the card edges horizontally
            rotated_id_card = cv2.rotate(
                rotated_id_card, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if line_angle < 45:
                rotated_id_card = cv2.rotate(
                    rotated_id_card, cv2.ROTATE_90_CLOCKWISE)

    # Resize the rotated ID card (adjust the dimensions as needed)
    resized_id_card = cv2.resize(rotated_id_card, (100, 40))
    cut_image = rotated_id_card[h-6000:h-300]
    # Save the cropped, rotated, and resized image
    cv2.imwrite(
        f'{desktop_path}\id_card_cropped_rotated_resized.jpg', cut_image)
    cv2.imwrite(f'{desktop_path}\id_card_cropped_rotated.jpg', rotated_id_card)
    print("Done")

    # Display the cropped, rotated, and resized image

else:
    print("No ID card found in the image.")
