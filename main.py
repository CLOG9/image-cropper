# import cv2
# import numpy as np

# def main():
#     # Load the image
#     input_path = "./0.jpg"
#     image = cv2.imread(input_path)
#     h = image.shape[0]
#     w= image.shape[1]
#     target_resolution = (round(w* 50 /100) , round(h * 50 /100))
#     image =  cv2.resize(image, target_resolution)
#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     output_path = "./gray_image.jpg"
#     cv2.imwrite(output_path, gray_image)
#     # Apply Gaussian blur
#     blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#     output_path = "./gaussian_image.jpg"
#     cv2.imwrite(output_path, blurred_image)


#     edged = cv2.Canny(blurred_image, 10, 20)
#     output_path = "./canny_image.jpg"
#     cv2.imwrite(output_path, edged)
#     # Use the Sobel operator to find the gradient magnitude and direction
#     sobel_x = cv2.Sobel(edged, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(edged, cv2.CV_64F, 0, 1, ksize=3)
#     gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
#     gradient_direction = np.arctan2(sobel_y, sobel_x)

#     # Use the Hough transformation to find lines
#     lines = cv2.HoughLines(gradient_magnitude.astype(np.uint8), 1, np.pi / 180, threshold=100)
    
#     # Draw the lines on the original image
#     result_image = image.copy()
#     for line in lines:
#         rho, theta = line[0]
#         a, b = np.cos(theta), np.sin(theta)
#         x0, y0 = a * rho, b * rho
#         x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
#         x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
#         cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     # Save the result in the current working directory
#     output_path = "./output_image.jpg"
#     cv2.imwrite(output_path, result_image)

#     # Display the result
#     cv2.imshow("Result", result_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

#####################################################


# import cv2
# import numpy as np

# def main():
#     # Load the image
#     input_path = "./0.jpg"
#     image = cv2.imread(input_path)

#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian blur
#     blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#     # Use the Sobel operator to find the gradient magnitude
#     gradient_magnitude = cv2.magnitude(cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3),
#                                        cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3))
    
#     # Use the Hough transformation to find lines (probabilistic version)
#     lines = cv2.HoughLinesP(gradient_magnitude.astype(np.uint8), 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
#     print(len(lines))
#     # Filter lines based on length (you can adjust the length threshold)
#     filtered_lines = non_max_suppression(lines, threshold=100)
#     print(len(filtered_lines))
#     # Draw the lines on the original image
#     result_image = image.copy()
#     for line in filtered_lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

#     # Save the result in the current working directory
#     output_path = "./output_image.jpg"
#     cv2.imwrite(output_path, result_image)

#     # Display the result
#     cv2.imshow("Result", result_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def non_max_suppression(lines, threshold):
#     if len(lines) == 0:
#         return []

#     # Convert lines to numpy array
#     lines = np.array(lines).reshape(-1, 4)

#     # Calculate line lengths
#     lengths = np.linalg.norm(lines[:, :2] - lines[:, 2:], axis=1)

#     # Get indices to keep using a threshold
#     indices_to_keep = lengths > threshold

#     # Filter lines based on the threshold
#     filtered_lines = lines[indices_to_keep]

#     return filtered_lines.reshape(-1, 1, 4).tolist()

# if __name__ == "__main__":
#     main()

## didnt find the quads yet


####################################################################################


import cv2
import numpy as np
import math

def main():
    # Load the image
    input_path = "./0.jpg"
    image = cv2.imread(input_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Use the Sobel operator to find the gradient magnitude
    gradient_magnitude = cv2.magnitude(cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3),
                                       cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3))
    
    # Use the Hough transformation to find lines (probabilistic version)
    lines = cv2.HoughLinesP(gradient_magnitude.astype(np.uint8), 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    print(len(lines))
    # Filter lines based on length (you can adjust the length threshold)
    filtered_lines = non_max_suppression(lines, threshold=100)
    print(len(filtered_lines))

    best_quadrilateral = find_best_quadrilateral(filtered_lines)
    # Draw the lines on the original image
    result_image = image.copy()
    for line in best_quadrilateral:
        x1, y1, x2, y2 = line[0]
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Save the result in the current working directory
    output_path = "./output_image.jpg"
    cv2.imwrite(output_path, result_image)

    # Display the result
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def non_max_suppression(lines, threshold):
    if len(lines) == 0:
        return []

    # Convert lines to numpy array
    lines = np.array(lines).reshape(-1, 4)

    # Calculate line lengths
    lengths = np.linalg.norm(lines[:, :2] - lines[:, 2:], axis=1)

    # Get indices to keep using a threshold
    indices_to_keep = lengths > threshold

    # Filter lines based on the threshold
    filtered_lines = lines[indices_to_keep]

    return filtered_lines.reshape(-1, 1, 4).tolist()


def find_best_quadrilateral(lines):
    best_score = -1
    best_quadrilateral = []

    # Iterate through every combination of four lines
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            for k in range(j+1, len(lines)):
                for l in range(k+1, len(lines)):
                    quadrilateral = [lines[i][0], lines[j][0], lines[k][0], lines[l][0]]
                    score = heuristic_scoring_function(quadrilateral)

                    # Update the best quadrilateral if the score is higher
                    if score > best_score:
                        best_score = score
                        best_quadrilateral = quadrilateral

    return best_quadrilateral


def heuristic_scoring_function(quadrilateral):
    # Calculate angles between lines in the quadrilateral
    angles = []
    for i in range(4):
        p1 = np.array(quadrilateral[i][:2])
        p2 = np.array(quadrilateral[(i + 1) % 4][:2])
        p3 = np.array(quadrilateral[(i + 2) % 4][:2])

        vector1 = p1 - p2
        vector2 = p3 - p2

        dot_product = np.dot(vector1, vector2)
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)

        angle = math.degrees(math.acos(dot_product / norm_product))
        angles.append(angle)

    # Score based on the sum of angles
    score = sum(angles)

    return score

if __name__ == "__main__":
    main()