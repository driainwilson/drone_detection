import cv2
import numpy as np
from numpy import typing as npt

from loguru import logger

class VideoWriter:
    """
    A class to write a sequence of images to a video file.

    This class uses OpenCV to create a video from a series of images.
    The user initializes the class with video properties, adds frames one by one,
    and then calls the save method to finalize the video.
    """

    def __init__(self, filename: str, fourcc: str = 'mp4v', fps: float = 20.0, frame_size: tuple[int, int] = (640, 480)) -> None:
        """
        Initializes the VideoWriter object.

        Args:
            filename (str): The name of the output video file (e.g., 'output.mp4').
            fourcc (str): The 4-character code of the codec used to compress the frames.
                          Examples: 'mp4v' for .mp4, 'XVID' for .avi.
            fps (float): The frame rate of the created video stream.
            frame_size (tuple): A tuple of integers (width, height) for the frame size.
        """
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Filename must be a non-empty string.")
        if not isinstance(fourcc, str) or len(fourcc) != 4:
            raise ValueError("fourcc must be a 4-character string.")
        if not isinstance(fps, (int, float)) or fps <= 0:
            raise ValueError("FPS must be a positive number.")
        if not isinstance(frame_size, tuple) or len(frame_size) != 2 or not all(isinstance(i, int) and i > 0 for i in frame_size):
            raise ValueError("frame_size must be a tuple of two positive integers (width, height).")

        self.filename: str = filename
        self.fourcc_code: int = cv2.VideoWriter_fourcc(*fourcc)
        self.fps: float = float(fps)
        self.frame_size: tuple[int, int] = frame_size
        self.video_writer: cv2.VideoWriter = cv2.VideoWriter(self.filename, self.fourcc_code, self.fps, self.frame_size)

        if not self.video_writer.isOpened():
            raise IOError(f"Could not open video writer for filename: {self.filename}")

        self.frames_added: int = 0
        logger.debug(f"VideoWriter initialized for '{self.filename}' with size {self.frame_size} at {self.fps} FPS.")

    def add_frame(self, image: npt.NDArray[np.uint8]) -> None:
        """
        Adds a single image frame to the video.

        The image must be a NumPy array and match the frame_size specified
        during initialization.

        Args:
            image (NDArray[np.uint8]): The image to add as a frame. It should be a
                                       NumPy array of shape (height, width, 3) and
                                       dtype uint8.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a NumPy array.")

        # OpenCV expects (height, width) but frame_size is (width, height)
        expected_shape = (self.frame_size[1], self.frame_size[0], 3)
        if image.shape != expected_shape:
            raise ValueError(f"Image shape {image.shape} does not match the expected frame shape {expected_shape}.")

        self.video_writer.write(image)
        self.frames_added += 1

    def save(self) -> None:
        """
        Releases the video writer, saving and finalizing the video file.
        This method must be called when all frames have been added.
        """
        if self.video_writer.isOpened():
            self.video_writer.release()
            logger.info(f"Video saved as '{self.filename}'. A total of {self.frames_added} frames were written.")
        else:
            logger.debug("VideoWriter was already closed.")