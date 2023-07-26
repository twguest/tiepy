import numpy as np
import cv2 as cv


def match_template(templ, image, method=3):
    """
    Perform template matching using OpenCV's cv2.matchTemplate function.

    This function matches a given template (templ) to an input image (image) using the specified method.

    Parameters:
        templ (numpy.ndarray): The template image to be matched against the input image.
                               It should be a 2D array (grayscale image).
        image (numpy.ndarray): The input image to which the template will be matched.
                               It should be a 2D array (grayscale image).
        method (int, optional): The method to be used for matching.
                                Refer to cv2.matchTemplate documentation for method options.
                                Defaults to 3 (cv2.TM_CCOEFF_NORMED).

    Returns:
        numpy.ndarray: The result of template matching as a 2D array of floating-point values.
                       This array represents the similarity between the template and the input image
                       at each location.
        tuple: A tuple containing the location (x, y) of the best match in the input image.
               This location corresponds to the top-left corner of the matched region.

    Raises:
        AssertionError: If the 'method' parameter is not of integer type or outside the range [0, 5].

    Notes:
        - The template and input image should be grayscale images of the same dtype and have proper dimensions.
        - The output tuple (max_loc) represents the location of the best match in the input image.
    """
    assert isinstance(method, int) and 0 <= method <= 5, "method should be an integer index in the range [0, 5]"

    res = cv.matchTemplate(templ=templ, image=image, method=method)
    _, _, _, max_loc = cv.minMaxLoc(res)

    return res, max_loc
