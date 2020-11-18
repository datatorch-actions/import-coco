from datatorch.api.scripts.import_coco import import_coco
from datatorch import get_inputs

inputs = get_inputs()

if __name__ == "__main__":
    file_path: str = inputs.get("path")
    project: str = inputs.get("project")
    import_bbox: bool = inputs.get("bbox")
    import_segmentation: bool = inputs.get("segmentation")
    max_iou: float = inputs.get("maxIou")
    simplify: float = inputs.get("simplify")
    ignore_annotations_with_ids: bool = inputs.get("ignoreAnnotationsWithIds")

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
