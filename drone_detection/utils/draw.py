import cv2
import math
import numpy as np
from numpy import typing as npt


def draw_bbox(image: npt.NDArray[np.uint8], bbox_xyxy: tuple[float | int, ...],
              confidence: float | None = None, colour=(0, 255, 0),
              thickness=2) -> np.ndarray:
    """
    Draws a detection bounding box on an image.

    Args:
        image: The image to draw on.
        detection: The Detection object containing the bounding box and confidence.
        colour: The color of the bounding box.
        thickness: The thickness of the bounding box lines.

    Returns:
        The image with the bounding box drawn on it.
    """

    x1, y1, x2, y2 = np.rint(bbox_xyxy).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)

    if confidence is not None:
        label = f"{confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thickness)

    return image


def draw_track(image: np.ndarray,
               bbox_xyxy: tuple[float | int, ...], track_id: int,
               velocity_xy: tuple[float, float] | None = None,
               direction_radians: float | None = None,
               color=(0, 255, 0),
               thickness=1) -> np.ndarray:
    """
    Draws a detection bounding box, velocity vector, and direction indicator on an image.

    Args:
        image: The image to draw on.
        bbox_xyxy: The Detection object containing the bounding box and confidence.
        velocity_xy: The velocity vector (vx, vy).
        direction_radians: The direction in radians.
        color: The color of the bounding box and velocity vector.
        thickness: The thickness of the bounding box lines and velocity vector.

    Returns:
        The image with the bounding box, velocity vector, and direction indicator drawn on it.
    """

    x1, y1, x2, y2 = np.rint(bbox_xyxy).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    label = f"[{track_id}]"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness, lineType=cv2.LINE_AA)

    # Draw velocity vector
    if velocity_xy is not None:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        vx, vy = velocity_xy
        arrow_length = 1  # Adjust as needed
        arrow_x = int(cx + vx * arrow_length)
        arrow_y = int(cy + vy * arrow_length)
        cv2.arrowedLine(image, (cx, cy), (arrow_x, arrow_y), color, thickness)

    if direction_radians is not None:
        # Draw direction indicator (small line)
        direction_length = 15
        direction_x = int(cx + direction_length * math.cos(direction_radians))
        direction_y = int(cy + direction_length * math.sin(direction_radians))
        cv2.line(image, (cx, cy), (direction_x, direction_y), color, thickness)

    return image


def draw_classification(image: np.ndarray, bbox_xyxy: tuple[int, ...], classifications: dict[str, float],
                        color=(0, 255, 0), thickness=1):
    """
    Draws a dictionary of classifications as horizontal bars, with length proportional to probability.

    Args:
        image: The image to draw on.
        bbox_xyxy: Bounding box coordinates (xmin, ymin, xmax, ymax).
        classifications: A dictionary of classifications and probabilities.
        color: The color of the bars.
        thickness: The thickness of the bar lines.

    Returns:
        The image with the classifications drawn on it.
    """

    xmin, ymin, xmax, ymax = bbox_xyxy
    x = int(xmax) + 5  # Right edge of the bbox, with a 5-pixel padding
    y = int(ymin)
    dy = 9 # Vertical space between lines
    bar_height = 4  # Height of the bar
    max_bar_length = 50  # Maximum length of the bar (pixels)
    font_size = 0.3
    for class_name, probability in classifications.items():
        label = f"{class_name[0]}"
        # Get the width of the text label
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
        text_width = text_size[0]

        # Position the text
        text_x = x
        text_y = y + bar_height  # Align text with the top of the bar
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness,
                    lineType=cv2.LINE_AA)

        # Calculate bar length based on probability
        bar_length = int(probability * max_bar_length)

        # Draw the bar to the right of the text
        bar_start = (text_x + text_width + 5, y)  # 5 pixels spacing between text and bar
        bar_end = (text_x + text_width + 5 + bar_length, y + bar_height)
        cv2.rectangle(image, bar_start, bar_end, color, thickness=cv2.FILLED)

        # Increment y for the next bar, including bar height and spacing
        y += dy

    return image

def draw_classification_old(image: np.ndarray, bbox_xyxy: tuple[int, ...], classifications: dict[str, float],
                        color=(0, 255, 0), thickness=1):
    """
    Draws a dictionary of classifications and probabilities at the top right of the image.

    Args:
        image: The image to draw on.
        classifications: A dictionary of classifications and probabilities.
        color: The color of the text.
        thickness: The thickness of the text.

    Returns:
        The image with the classifications drawn on it.
    """

    xmin, ymin, xmax, ymax = bbox_xyxy
    x = int(xmax) + 5  # Right edge of the bbox, with a 5-pixel padding
    y = int(ymin)
    dy = 10  # Vertical space between lines

    for class_name, probability in classifications.items():
        label = f"{class_name}: {probability:.2f}"
        # text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
        # text_x = x - text_size[0]  # Right-align the text
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, thickness, lineType=cv2.LINE_AA)
        y += dy

    return image


def draw_state(image: np.ndarray, state: dict[str, float], color=(255, 0, 0), thickness=1):
    """
    Draws a dictionary of classifications and probabilities at the top right of the image.

    Args:
        image: The image to draw on.
        classifications: A dictionary of classifications and probabilities.
        color: The color of the text.
        thickness: The thickness of the text.

    Returns:
        The image with the classifications drawn on it.
    """
    x = 20  # left edge of the image, with a 10-pixel padding
    y = 20  # Top of the image, with a 20-pixel padding
    dy = 20  # Vertical space between lines

    for class_name, value in state.items():
        label = f"{class_name}: {value:.2f}"
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, lineType=cv2.LINE_AA)
        y += dy

    return image


def draw_threat_scores(
    image: npt.NDArray[np.uint8],
    scores: dict[int, float],
    font_scale: float = 0.5,
    thickness: int = 1,
) -> npt.NDArray[np.uint8]:
    """
    Draws multiple threat scores in colored boxes at the top-left of an image, stacked vertically.
    The box color scales from green to red based on the score, and the text is white.

    Args:
        image (np.ndarray): The input image (OpenCV format, BGR).
        scores (dict[str, float]): A dictionary of threat scores, where keys are labels and values are scores (0-100).
        font_scale (float): The scale of the font.
        thickness (int): The thickness of the font.

    Returns:
        np.ndarray: The image with the threat score boxes drawn on it.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 10  # Margin from the image borders
    padding = 10  # Padding inside the box
    y_offset = margin  # Initial Y offset for the first box

    for label, score in scores.items():
        # 1. Validate the score
        score = max(0, min(100, score))

        # 2. Calculate the box color
        normalized_score = score / 100.0
        red_value = int(255 * normalized_score)
        green_value = int(255 * (1 - normalized_score))
        box_color = (0, green_value, red_value)
        text_color = (255, 255, 255)  # White color for the text

        # 3. Prepare the text and calculate box size
        text = f"[{label}] Threat Score: {int(round(score))}"

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # 4. Define box and text positions
        # Box top-left corner
        box_tl = (margin, y_offset)
        # Box bottom-right corner
        box_br = (
            margin + text_width + padding * 2,
            y_offset + text_height + padding * 2,
        )
        # Text bottom-left corner (inside the box)
        text_bl = (margin + padding, y_offset + text_height + padding)

        # 5. Draw the filled box and the text
        cv2.rectangle(
            img=image,
            pt1=box_tl,
            pt2=box_br,
            color=box_color,
            thickness=2,  # Use FILLED for a solid box
        )

        cv2.putText(
            img=image,
            text=text,
            org=text_bl,
            fontFace=font,
            fontScale=font_scale,
            color=text_color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        # Update the Y offset for the next box
        y_offset = box_br[1] + margin  # Add margin between boxes

    return image
