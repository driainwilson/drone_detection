def xyxy_to_xywh(bbox: tuple[float | int, ...]) -> tuple[float | int, ...]:
    """
    Convert bounding box format from [x1, y1, x2, y2] to [x, y, width, height].

    Args:
        bbox (tuple): Bounding box in [x1, y1, x2, y2] format.

    Returns:
        tuple: Bounding box in [x, y, width, height] format.
    """
    x1, y1, x2, y2 = bbox
    x = x1
    y = y1
    width = x2 - x1
    height = y2 - y1
    return x, y, width, height


def xywh_to_xyxy(bbox: tuple[float, ...]) -> tuple[float, ...]:
    """
    Convert bounding box format from [x, y, width, height] to [x1, y1, x2, y2].

    Args:
        bbox (tuple): Bounding box in [x, y, width, height] format.

    Returns:
        tuple: Bounding box in [x1, y1, x2, y2] format.
    """
    x, y, width, height = bbox
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height
    return x1, y1, x2, y2


def xyxy_to_cxcywh(bbox: tuple[float, ...]) -> tuple[float, ...]:
    """
    Convert bounding box format from [x1, y1, x2, y2] to [center_x, center_y, width, height]

    Args:
        bbox (tuple): Bounding box in [x1, y1, x2, y2] format.

    Returns:
        tuple: Bounding box in [center_x, center_y, width, height] format (normalized).
    """
    x1, y1, x2, y2 = bbox
    center_x = ((x1 + x2) / 2)
    center_y = ((y1 + y2) / 2)
    width = (x2 - x1)
    height = (y2 - y1)
    return center_x, center_y, width, height


def cxcywh_to_xyxy(bbox: tuple[float | int, ...]) -> tuple[float | int, ...]:
    """
    Convert bounding box format from [center_x, center_y, width, height]  to [x1, y1, x2, y2].

    Args:
        bbox (tuple): Bounding box in [center_x, center_y, width, height] format (normalized).

    Returns:
        tuple: Bounding box in [x1, y1, x2, y2] format.
    """
    center_x, center_y, width, height = bbox
    x1 = (center_x - width / 2)
    y1 = (center_y - height / 2)
    x2 = (center_x + width / 2)
    y2 = (center_y + height / 2)
    return x1, y1, x2, y2
