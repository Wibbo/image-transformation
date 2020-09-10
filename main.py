import cv2
import numpy as np
from math import hypot


def get_line_length(point_array, point1, point2):
    """
    Determines the length between two specified points in the supplied corner array.
    :param point_array: The corner array to process.
    :param point1: The first point in the corner array.
    :param point2: The second point in the corner array.
    :return: The distance between the two points.
    """
    x1 = point_array[point1][0]
    x2 = point_array[point2][0]
    y1 = point_array[point1][1]
    y2 = point_array[point2][1]

    x = x2 - x1
    y = y2 - y1

    return hypot(x, y)


def get_maximum_width(point_array):
    """
    Returns the greater distance between the two top and two bottom points.
    :param point_array: Contains a list of coordinates for the four corners.
    :return: The distance between the top and bottom points whichever is greater.
    """
    width1 = get_line_length(point_array, 0, 1)
    width2 = get_line_length(point_array, 2, 3)
    return max(width1, width2)


def get_maximum_height(point_array):
    """
    Returns the greater distance between the two left and two right points.
    :param point_array: Contains a list of coordinates for the four corners.
    :return: The distance between the left and right points whichever is greater.
    """
    height1 = get_line_length(point_array, 0, 2)
    height2 = get_line_length(point_array, 1, 3)
    return max(height1, height2)


def get_end_image_corners(start_corners):
    """
    Returns the end coordinates for the corners of the image.
    :param start_corners:
    :return:
    """
    width = get_maximum_width(start_corners)
    height = get_maximum_height(start_corners)
    offset = 30

    corners = ([(0 + offset, 0 + offset), (width-1 + offset, 0 + offset),
                (0 + offset, height-1 + offset), (width-1 + offset, height-1 + offset)])
    return corners, height, width


def set_text(source_image, coord_list):
    """
    Adds labels to each corner of the image.
    :param source_image: The image on which to add labels.
    :param coord_list: A list of coordinates for the label positions.
    :return: Nothing.
    """
    font = cv2.FONT_HERSHEY_PLAIN
    text_colour = (0, 145, 255)

    for index, coord in enumerate(coord_list):
        corner_label = 'C' + str(index + 1)
        cv2.putText(source_image, corner_label, tuple(coord), font, 2, text_colour, 2, cv2.LINE_AA)


def get_start_image_corners(source_image, image_contours):
    """
    Returns the start coordinates for the corners of the image.
    :param source_image:
    :param image_contours:
    :return:
    """
    epsilon = 0.02 * cv2.arcLength(image_contours, True)
    approx_corners = cv2.approxPolyDP(image_contours, epsilon, True)
    cv2.drawContours(source_image, approx_corners, -1, (163, 255, 119), 20)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())

    # Rearrange the order of the corner points to prepare for subsequent processing.
    # This can be improved and you may need to set the appropriate order based on
    # how the corners are initially ordered.
    approx_corners = [approx_corners[i] for i in [0, 2, 1, 3]]
    # approx_corners = [approx_corners[i] for i in [0, 3, 2, 1]]
    return source_image, approx_corners


def load_image(image_path, width, height):
    """
    Loads an image from the specified file and changes its size.
    :param image_path: The relative path & name of the image file.
    :param width: The required width of the image.
    :param height: The required height of the image.
    :return: A re-sized object representing the image.
    """
    img = cv2.imread(image_path)
    return cv2.resize(img, (width, height))


def create_binary_image(image):
    """
    Creates a binary threshold of the image by
    classifying each pixel as either black or white.
    :param image: The image to be processed.
    :return: A classified image.
    """
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, new_image = cv2.threshold(new_image, 225, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    return new_image


def transform_image(source_image, start_pos, end_pos):
    """
    This is where the magic happens. OpenCV provides the 2 functions required to
    transform the image once the start and end corner coordinates are provided.
    :param source_image:
    :param start_pos:
    :param end_pos:
    :return:
    """
    h, w = source_image.shape[:2]
    homography, _ = cv2.findHomography(start_pos, end_pos, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    transform = cv2.warpPerspective(source_image, homography, (w, h), flags=cv2.INTER_LINEAR)
    return transform


# The main process flow starts here.
# Load an image and resize it.
test_image = load_image('card3.jpg', 800, 600)

# Copy our image so we can display progress.
contour_image_1 = test_image.copy()
contour_image_2 = test_image.copy()
corner_image = test_image.copy()
threshold_image = create_binary_image(contour_image_1)

# Get an array or contours from the binary image we previously created.
contours, _ = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_restricted = sorted(contours, key=cv2.contourArea, reverse=True)[0]

# Create images including contours and restricted contours.
cv2.drawContours(contour_image_1, contours, -1, (0, 255, 0), 3)
cv2.drawContours(contour_image_2, contours_restricted, -1, (0, 255, 0), 3)
image_with_corners, corner_array = get_start_image_corners(corner_image, contours_restricted)
end_corners, end_height, end_width = get_end_image_corners(corner_array)
transformed_image = transform_image(test_image, np.float32(corner_array), np.float32(end_corners))

# Display all interim and final images.
cv2.imshow('1: The original image', test_image)
cv2.imshow('2: A threshold version of the image', threshold_image)
cv2.imshow('3: The original image with contours added', contour_image_1)
cv2.imshow('4: The original image with filtered contours', contour_image_2)
set_text(image_with_corners, corner_array)
cv2.imshow('5: The image with corners marked', image_with_corners)
cv2.imshow('6: The transformed image', transformed_image)

# Print basic details of the transformation.
print(f'Top line has a length of {get_line_length(corner_array, 0, 1)}')
print(f'Bottom line has a length of {get_line_length(corner_array, 2, 3)}')
print(f'Left line has a length of {get_line_length(corner_array, 0, 2)}')
print(f'Right line has a length of {get_line_length(corner_array, 1, 3)}')
print(f'Greatest width is {get_maximum_width(corner_array)}')
print(f'Greatest height is {get_maximum_height(corner_array)}')

cv2.waitKey(0)
cv2.destroyAllWindows()

















