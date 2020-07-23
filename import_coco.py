from datatorch.api.scripts.import_coco import import_coco
from datatorch import get_inputs

if __name__ == "__main__":
    file_path: str = get_inputs("path")
    project: str = get_inputs("project")
    import_bbox: bool = get_inputs("bbox")
    import_segmentation: bool = get_inputs("segmentation")
    max_iou: float = get_inputs("maxIou")
    simplify: float = get_inputs("simplify")
    ignore_annotations_with_ids: bool = get_inputs("ignoreAnnotationsWithIds")

    # Call the importer already built into the python SDK at
    # `datatorch import coco <args>`
    import_coco(
        file_path,
        project,
        import_bbox=import_bbox,
        import_segmentation=import_segmentation,
        max_iou=max_iou,
        simplify_tolerance=simplify,
        ignore_annotations_with_ids=ignore_annotations_with_ids
    )
