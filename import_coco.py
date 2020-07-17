import os
import sys
from sys import path
import numpy as np
from typing import Counter, List
from pycocotools.coco import COCO

from datatorch import get_inputs
from datatorch.api import Annotation, ApiClient, BoundingBox, File, Segmentations, Where


def format_segmentation(segmentation: List[List[int]]) -> List[List[List[int]]]:
    return [np.reshape(polygon, (-1, 2)).tolist() for polygon in segmentation]


def bbox_iou(bb1o: BoundingBox, bb2o: BoundingBox):
    """ Calculate the Intersection over Union (IoU) of two bounding boxes. """
    bb1 = {
        "x1": bb1o.x,
        "y1": bb1o.y,
        "x2": bb1o.bottom_right[0],
        "y2": bb1o.bottom_right[1],
    }
    bb2 = {
        "x1": bb2o.x,
        "y1": bb2o.y,
        "x2": bb2o.bottom_right[0],
        "y2": bb2o.bottom_right[1],
    }

    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    return intersection_area / float(bb1_area + bb2_area - intersection_area)


def has_bbox(bbox: BoundingBox, bboxs: List[BoundingBox], max_iou: float) -> bool:
    for bb in bboxs:
        if bbox_iou(bbox, BoundingBox.xywh(*bb)) > max_iou:
            return True
    return False


def mask_iou(mask1, mask2):
    union = mask1 * mask2
    intersect = mask1 + mask2
    return np.count_nonzero(union) / np.count_nonzero(intersect)


def has_mask(coco_anno: dict, segments: list, max_iou: float) -> bool:
    """ 
    Checks if the coco annotation has a matching annotation in the segmentation
    list. Returns true if the iou of any segmentation in the list is over the
    max iou otherwise false.    
    """
    anno_mask = coco.annToMask(coco_anno)
    # Check if segmentation already exists.
    for segment in segments:
        anno["segmentation"] = [np.array(polygon).flatten() for polygon in segment]
        dt_mask = coco.annToMask(anno)
        if mask_iou(anno_mask, dt_mask) > max_iou:
            return True
    return False


if __name__ == "__main__":
    file_path: str = get_inputs("path")
    project_id: str = get_inputs("projectId")
    perserve_ids: bool = get_inputs("preserveIds")
    import_bbox: bool = get_inputs("bbox")
    import_segmentation: bool = get_inputs("segmentation")
    max_iou: float = get_inputs("maxIou")
    check_iou: bool = max_iou != -1.0

    if not import_segmentation and not import_bbox:
        print("Nothing to import. Both segmentation and bbox are disabled.")
        sys.exit(1)

    if not os.path.isfile(file_path):
        print("Provided path is not a file.")
        sys.exit(2)

    # Get DataTorch project information
    api = ApiClient()
    project = api.project(project_id)
    labels = project.labels()
    names_mapping = dict(((label.name, label) for label in labels))

    # Load coco file
    coco = COCO(file_path)
    coco_categories = coco.loadCats(coco.getCatIds())

    # Maps COCO Category ids to DataTorch ids.
    # This is done by matching label names.
    label_mapping: dict = {}
    for category in coco_categories:
        name = category["name"]
        label_mapping[category["id"]] = datatorch_label = names_mapping.get(name)

        if not datatorch_label:
            print(f"Could not find {name} in project labels.")
            continue

    # Iterate each annotations to add them to
    # datatorch
    coco_category_ids = label_mapping.keys()
    for image_id in coco.getImgIds():
        (coco_image,) = coco.loadImgs(image_id)
        image_name = coco_image["file_name"]

        file_filter = Where(name=image_name)
        dt_files = project.files(file_filter, limit=2)

        if len(dt_files) > 1:
            print(f"Multiple files found of {image_name}, skipping")
            continue

        if len(dt_files) == 0:
            print(f"No files found of {image_name}, skipping")
            continue

        dt_file: File = dt_files[0]

        dt_segmentations = []
        dt_bbox = []
        if check_iou:
            for anno in dt_file.annotations:
                for source in anno.sources:
                    if source.type == "PaperSegmentations":
                        dt_segmentations.append(source.path_data)
                    if source.type == "PaperBox":
                        dt_bbox.append(
                            (source.x, source.y, source.width, source.height,)
                        )

        coco_annotation_ids = coco.getAnnIds(
            catIds=coco_category_ids, imgIds=coco_image["id"]
        )
        coco_annotations = coco.loadAnns(ids=coco_annotation_ids)

        imported_count = 0
        for anno in coco_annotations:
            if anno.get("datatorch_id") is not None:
                continue

            label = label_mapping.get(anno["category_id"])
            if label is None:
                continue

            dt_anno = Annotation(label=label)
            created_bbox = False
            created_segmentation = False

            if import_bbox:
                bbox = BoundingBox.xywh(*anno["bbox"])
                if not check_iou or (
                    check_iou and not has_bbox(bbox, dt_bbox, max_iou)
                ):
                    imported_count += 1
                    created_bbox = True
                    dt_anno.add(bbox)

            # If the bbox was suppose to be created but wasn't, no point in
            # checking segmentation since it probably will be the same.
            if import_bbox and not created_bbox:
                continue

            if import_segmentation:
                path_data = format_segmentation(anno["segmentation"])

                if not check_iou or (
                    check_iou and not has_mask(anno, dt_segmentations, max_iou)
                ):
                    segment = Segmentations(path_data=path_data)
                    created_segmentation = True
                    imported_count += 1
                    dt_anno.add(segment)

            if created_segmentation or created_bbox:
                dt_file.add(dt_anno)

        print(f"Added {imported_count} annnotations to {dt_file.name}.")
