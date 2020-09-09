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


def get_end_image_corners(start_corners, end_corners):
    w = get_maximum_width(start_corners)
    h = get_maximum_height(end_corners)

    corners = ([(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)])
    return corners, h,w


def get_start_image_corners(source_image, image_contours):
    epsilon = 0.02 * cv2.arcLength(image_contours, True)
    approx_corners = cv2.approxPolyDP(image_contours, epsilon, True)
    cv2.drawContours(source_image, approx_corners, -1, (163, 255, 119), 20)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())

    for index, c in enumerate(approx_corners):
        character = chr(65 + index)
        cv2.putText(source_image, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2. LINE_8)

    # Rearranging the order of the corner points
    approx_corners = [approx_corners[i] for i in [0, 2, 1, 3]]
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
    _, new_image = cv2.threshold(new_image, 225, 255, cv2.THRESH_BINARY_INV)
    return new_image


def transform_image(source_image, start_corners, end_corners):

    h, w = source_image.shape[:2]
    H, _ = cv2.findHomography(start_corners, end_corners, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    print('\nThe homography matrix is: \n', H)
    un_warped = cv2.warpPerspective(source_image, H, (w, h), flags=cv2.INTER_LINEAR)

    # plot

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    # f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(source_image)
    ax1.set_title('Original Image')

    x = [start_corners[0][0], start_corners[2][0], start_corners[3][0], start_corners[1][0], start_corners[0][0]]
    y = [start_corners[0][1], start_corners[2][1], start_corners[3][1], start_corners[1][1], start_corners[0][1]]

    ax2.imshow(source_image)
    ax2.plot(x, y, color='yellow', linewidth=3)
    ax2.set_ylim([h, 0])
    ax2.set_xlim([0, w])
    ax2.set_title('Target Area')

    plt.show()
    return un_warped


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
image_corners, corner_array = get_start_image_corners(corner_image, contours_restricted)

# x = transform_image(test_image, corner_array, )

# Display all interim and final images.
cv2.imshow('1: The original image', test_image)
cv2.imshow('2: A threshold version of the image', threshold_image)
cv2.imshow('3: The original image with contours added', contour_image_1)
cv2.imshow('4: The original image with filtered contours', contour_image_2)
cv2.imshow('5: The image with corners marked', image_corners)

# Print basic details of the transformation.
print(f'Top line has a length of {get_line_length(corner_array, 0, 1)}')
print(f'Bottom line has a length of {get_line_length(corner_array, 2, 3)}')
print(f'Left line has a length of {get_line_length(corner_array, 0, 2)}')
print(f'Right line has a length of {get_line_length(corner_array, 1, 3)}')
print(f'Greatest width is {get_maximum_width(corner_array)}')
print(f'Greatest height is {get_maximum_height(corner_array)}')

cv2.waitKey(0)
cv2.destroyAllWindows()

















