import xml.etree.ElementTree as ET
import shutil  # For file copying
from typing import Dict, List, Tuple, Optional
from pathlib import Path  # Using pathlib for more robust path handling
from loguru import logger
import sys  # For configuring loguru

# --- Loguru Configuration ---
# Remove default handler to prevent duplicate console outputs if script is run multiple times
logger.remove()
# Add a new handler with a specific format and level
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")


def convert_voc_to_yolo(
        xml_file_path: Path,
        output_labels_dir: Path,
        class_mapping: Dict[str, int]
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Converts a single PASCAL VOC XML annotation file to YOLO format and saves it.

    Args:
        xml_file_path (Path): Path to the PASCAL VOC XML file.
        output_labels_dir (Path): Directory to save the YOLO format .txt label file.
        class_mapping (Dict[str, int]): A dictionary mapping class names to class IDs.

    Returns:
        Tuple[Optional[Path], Optional[str]]:
            - Path to the created .txt label file if successful, else None.
            - The image filename extracted from the XML, else None.
    """
    image_filename_from_xml: Optional[str] = None
    try:
        tree: ET.ElementTree = ET.parse(str(xml_file_path))
        root: ET.Element = tree.getroot()

        image_filename_element: Optional[ET.Element] = root.find('filename')
        if image_filename_element is None or image_filename_element.text is None:
            logger.warning(f"'filename' tag not found or empty in {xml_file_path}. Skipping.")
            return None, None
        image_filename_from_xml = image_filename_element.text

        size_element: Optional[ET.Element] = root.find('size')
        if size_element is None:
            logger.warning(f"'size' tag not found in {xml_file_path} (image: {image_filename_from_xml}). Skipping.")
            return None, image_filename_from_xml

        img_width_element: Optional[ET.Element] = size_element.find('width')
        img_height_element: Optional[ET.Element] = size_element.find('height')

        if img_width_element is None or img_width_element.text is None or \
                img_height_element is None or img_height_element.text is None:
            logger.warning(
                f"Image width or height tags are missing or empty in {xml_file_path} (image: {image_filename_from_xml}). Skipping.")
            return None, image_filename_from_xml

        img_width: int = int(img_width_element.text)
        img_height: int = int(img_height_element.text)

        if img_width == 0 or img_height == 0:
            logger.warning(
                f"Image width or height is 0 in {xml_file_path} (Width: {img_width}, Height: {img_height}, image: {image_filename_from_xml}). Skipping.")
            return None, image_filename_from_xml

        yolo_annotations: List[str] = []
        for obj_element in root.findall('object'):
            class_name_element: Optional[ET.Element] = obj_element.find('name')
            if class_name_element is None or class_name_element.text is None:
                logger.warning(
                    f"Object 'name' tag not found or empty in an object in {xml_file_path} (image: {image_filename_from_xml}). Skipping object.")
                continue
            class_name: str = class_name_element.text

            if class_name not in class_mapping:
                logger.warning(
                    f"Class '{class_name}' not in class_mapping. Skipping object in {xml_file_path} (image: {image_filename_from_xml}).")
                continue
            class_id: int = class_mapping[class_name]

            bndbox_element: Optional[ET.Element] = obj_element.find('bndbox')
            if bndbox_element is None:
                logger.warning(
                    f"Object 'bndbox' tag not found for class '{class_name}' in {xml_file_path} (image: {image_filename_from_xml}). Skipping object.")
                continue

            try:
                xmin: float = float(bndbox_element.findtext('xmin', '0'))
                ymin: float = float(bndbox_element.findtext('ymin', '0'))
                xmax: float = float(bndbox_element.findtext('xmax', '0'))
                ymax: float = float(bndbox_element.findtext('ymax', '0'))
            except (ValueError, TypeError) as e:
                logger.error(
                    f"Could not parse bounding box coordinates for an object in {xml_file_path} (image: {image_filename_from_xml}). Error: {e}. Skipping object.")
                continue

            if xmax <= xmin or ymax <= ymin:
                logger.warning(
                    f"Invalid bounding box (xmax <= xmin or ymax <= ymin) for class '{class_name}' in {xml_file_path} (image: {image_filename_from_xml}): "
                    f"xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}. Skipping object.")
                continue

            x_center: float = (xmin + xmax) / 2.0
            y_center: float = (ymin + ymax) / 2.0
            width: float = xmax - xmin
            height: float = ymax - ymin

            x_center_norm: float = x_center / img_width
            y_center_norm: float = y_center / img_height
            width_norm: float = width / img_width
            height_norm: float = height / img_height

            x_center_norm = max(0.0, min(1.0, x_center_norm))
            y_center_norm = max(0.0, min(1.0, y_center_norm))
            width_norm = max(0.0, min(1.0, width_norm))
            height_norm = max(0.0, min(1.0, height_norm))

            yolo_annotations.append(
                f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

        if not yolo_annotations:
            logger.info(
                f"No valid objects found or converted for {xml_file_path.name} (image: {image_filename_from_xml}). No label file created.")
            return None, image_filename_from_xml

        output_labels_dir.mkdir(parents=True, exist_ok=True)
        base_filename_stem: str = Path(image_filename_from_xml).stem  # e.g., "VS_P65" from "VS_P65.jpg"
        output_label_file_path: Path = output_labels_dir / f"{base_filename_stem}.txt"

        with open(output_label_file_path, 'w') as f:
            for ann in yolo_annotations:
                f.write(ann + '\n')

        logger.info(
            f"Successfully converted {xml_file_path.name} to {output_label_file_path.name} (image: {image_filename_from_xml})")
        return output_label_file_path, image_filename_from_xml

    except ET.ParseError:
        logger.error(f"Could not parse XML file {xml_file_path}. It might be corrupted or not a valid XML.")
        return None, None  # No image filename if XML can't be parsed
    except FileNotFoundError:
        logger.error(f"XML file not found at {xml_file_path}.")
        return None, None
    except Exception as e:
        logger.exception(f"An unexpected error occurred while processing {xml_file_path}: {e}")
        return None, image_filename_from_xml  # Return filename if extracted before error


def process_dataset_folders(
        image_input_dir: Path,
        annotation_input_dir: Path,
        yolo_base_output_dir: Path,
        class_mapping: Dict[str, int]
) -> None:
    """
    Processes folders of images and PASCAL VOC XML annotations,
    converts annotations to YOLO format, and organizes images and labels
    into a YOLO dataset structure.

    Args:
        image_input_dir (Path): Path to the folder containing source image files.
        annotation_input_dir (Path): Path to the folder containing PASCAL VOC XML annotation files.
        yolo_base_output_dir (Path): Base directory to save the structured YOLO dataset.
                                     Subdirectories 'images' and 'labels' will be created here.
        class_mapping (Dict[str, int]): Dictionary mapping class names to class IDs.
    """
    output_images_dir: Path = yolo_base_output_dir / 'images'
    output_labels_dir: Path = yolo_base_output_dir / 'labels'

    try:
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create output directories: {e}. Please check permissions and path.")
        return

    logger.info(f"Processing annotations from: {annotation_input_dir}")
    logger.info(f"Looking for corresponding images in: {image_input_dir}")
    logger.info(f"Outputting YOLO images to: {output_images_dir}")
    logger.info(f"Outputting YOLO labels to: {output_labels_dir}")

    if not annotation_input_dir.is_dir():
        logger.error(f"Annotation input directory not found: {annotation_input_dir}")
        return
    if not image_input_dir.is_dir():
        logger.error(f"Image input directory not found: {image_input_dir}")
        return

    xml_files: List[Path] = list(annotation_input_dir.glob('*.xml'))
    if not xml_files:
        logger.warning(f"No XML files found in '{annotation_input_dir}'.")
        return

    logger.info(f"Found {len(xml_files)} XML annotation files to process.")

    successful_conversions: int = 0
    image_copy_successes: int = 0
    image_not_found_count: int = 0
    label_created_img_missing_count: int = 0

    for xml_file_path in xml_files:
        logger.debug(f"Processing XML: {xml_file_path.name}")

        created_label_path, image_filename_in_xml = convert_voc_to_yolo(
            xml_file_path,
            output_labels_dir,
            class_mapping
        )

        if image_filename_in_xml is None:
            # Error during XML parsing or filename extraction, already logged by convert_voc_to_yolo
            continue

        source_image_path: Path = image_input_dir / image_filename_in_xml

        if not source_image_path.is_file():
            logger.warning(
                f"Image '{image_filename_in_xml}' (from XML '{xml_file_path.name}') not found at '{source_image_path}'.")
            image_not_found_count += 1
            if created_label_path:
                logger.info(
                    f"Label file '{created_label_path.name}' was created, but corresponding image was not found. The label file is kept.")
                label_created_img_missing_count += 1
            continue

        if created_label_path and created_label_path.exists():
            successful_conversions += 1
            target_image_path: Path = output_images_dir / image_filename_in_xml
            try:
                shutil.copy(str(source_image_path), str(target_image_path))
                logger.info(f"Copied image '{source_image_path.name}' to '{target_image_path}'")
                image_copy_successes += 1
            except Exception as e:
                logger.error(f"Failed to copy image '{source_image_path.name}' to '{target_image_path}'. Error: {e}")
        elif image_filename_in_xml:
            logger.info(
                f"Label file was not created for XML '{xml_file_path.name}' (image '{image_filename_in_xml}'). Image will not be copied.")

    logger.info("--- Processing Summary ---")
    logger.info(f"Total XML files processed: {len(xml_files)}")
    logger.info(f"Successfully converted annotations (label files created): {successful_conversions}")
    logger.info(f"Successfully copied images: {image_copy_successes}")
    logger.info(f"Source images not found for XMLs: {image_not_found_count}")
    if label_created_img_missing_count > 0:
        logger.warning(f"Label files created but corresponding image missing: {label_created_img_missing_count}")
    logger.info(f"Check '{output_images_dir}' for images and '{output_labels_dir}' for labels.")


if __name__ == '__main__':
    CLASS_MAPPING: Dict[str, int] = {
        'drone': 0,
    }

    base_example_dir = Path.home() / "data"
    base_example_dir.mkdir(exist_ok=True)
    source_images_folder: Path = base_example_dir / "DroneTestDataset" / "Drone_TestSet"
    source_annotations_folder: Path = base_example_dir / "DroneTestDataset" /"Drone_TestSet_XMLs"
    yolo_output_dataset_folder: Path = base_example_dir / "yolo_drone_test"


    logger.info("--- Starting dataset processing ---")
    process_dataset_folders(
        image_input_dir=source_images_folder,
        annotation_input_dir=source_annotations_folder,
        yolo_base_output_dir=yolo_output_dataset_folder,
        class_mapping=CLASS_MAPPING
    )

