import numpy as np
import supervisely_lib as sly
from supervisely_lib._utils import generate_free_name


def postprocess(state, ann: sly.Annotation, cur_meta: sly.ProjectMeta, res_meta: sly.ProjectMeta) \
        -> (sly.ProjectMeta, sly.Annotation):
    task_type = state["taskType"]
    if task_type == "seg":
        new_meta, new_ann = transform_for_segmentation(cur_meta, ann)
    elif task_type == "det":
        new_meta, new_ann = transform_for_detection(cur_meta, ann)
    elif task_type == "inst-seg":
        new_meta, new_ann = transform_for_instance_segmentation(cur_meta, ann)

    new_meta, res_meta, new_ann = merge_classes(new_meta, res_meta, new_ann)
    return res_meta, new_ann


def transform_for_detection(meta: sly.ProjectMeta, ann: sly.Annotation) -> (sly.ProjectMeta, sly.Annotation):
    new_classes = sly.ObjClassCollection()
    new_labels = []
    for label in ann.labels:
        new_class = label.obj_class.clone(name=label.obj_class.name + "-bbox", geometry_type=sly.Rectangle)
        if label.obj_class.geometry_type is sly.Rectangle:
            new_labels.append(label.clone(obj_class=new_class))
            if new_classes.get(new_class.name) is None:
                new_classes = new_classes.add(new_class)
        else:
            bbox = label.geometry.to_bbox()
            if new_classes.get(new_class.name) is None:
                new_classes = new_classes.add(new_class)
            new_labels.append(label.clone(bbox, new_class))
    res_meta = meta.clone(obj_classes=new_classes)
    res_ann = ann.clone(labels=new_labels)
    return (res_meta, res_ann)


def transform_for_segmentation(meta: sly.ProjectMeta, ann: sly.Annotation) -> (sly.ProjectMeta, sly.Annotation):
    new_classes = {}
    class_masks = {}
    for obj_class in meta.obj_classes:
        obj_class: sly.ObjClass
        new_class = obj_class.clone(name=obj_class.name + "-mask")
        new_classes[obj_class.name] = new_class
        class_masks[obj_class.name] = np.zeros(ann.img_size, np.uint8)

    new_class_collection = sly.ObjClassCollection(list(new_classes.values()))
    for label in ann.labels:
        label.draw(class_masks[label.obj_class.name], color=255)

    new_labels = []
    for class_name, white_mask in class_masks.items():
        mask = white_mask == 255
        obj_class = new_classes[class_name]
        bitmap = sly.Bitmap(data=mask)
        new_labels.append(sly.Label(geometry=bitmap, obj_class=obj_class))

    res_meta = meta.clone(obj_classes=new_class_collection)
    res_ann = ann.clone(labels=new_labels)
    return (res_meta, res_ann)


def transform_for_instance_segmentation(meta: sly.ProjectMeta, ann: sly.Annotation) -> (sly.ProjectMeta, sly.Annotation):
    new_classes = {}
    for obj_class in meta.obj_classes:
        obj_class: sly.ObjClass
        new_class = obj_class.clone(name=obj_class.name + "-mask")
        new_classes[obj_class.name] = new_class

    new_class_collection = sly.ObjClassCollection(list(new_classes.values()))
    new_labels = []
    for label in ann.labels:
        obj_class = new_classes[label.obj_class.name]
        new_labels.append(label.clone(obj_class=obj_class))

    res_meta = meta.clone(obj_classes=new_class_collection)
    res_ann = ann.clone(labels=new_labels)
    return (res_meta, res_ann)


def highlight_instances(meta: sly.ProjectMeta, ann: sly.Annotation) -> (sly.ProjectMeta, sly.Annotation):
    new_classes = []
    new_labels = []
    for idx, label in enumerate(ann.labels):
        new_cls = label.obj_class.clone(name=str(idx), color=sly.color.random_rgb())
        new_lbl = label.clone(obj_class=new_cls)

        new_classes.append(new_cls)
        new_labels.append(new_lbl)

    res_meta = meta.clone(obj_classes=sly.ObjClassCollection(new_classes))
    res_ann = ann.clone(labels=new_labels)
    return (res_meta, res_ann)


def merge_classes(cur_meta: sly.ProjectMeta, res_meta: sly.ProjectMeta, ann: sly.Annotation):
    existing_names = set([obj_class.name for obj_class in res_meta.obj_classes])
    mapping = {}
    for obj_class in cur_meta.obj_classes:
        obj_class: sly.ObjClass
        if obj_class.name in mapping:
            continue
        dest_class = res_meta.get_obj_class(obj_class.name)
        if dest_class is None:
            res_meta = res_meta.add_obj_class(obj_class)
            dest_class = obj_class
        elif obj_class != dest_class:
            new_name = generate_free_name(existing_names, obj_class.name)
            dest_class = obj_class.clone(name=new_name)
            res_meta = res_meta.add_obj_class(dest_class)
        mapping[obj_class.name] = dest_class

    new_labels = []
    for label in ann.labels:
        if label.obj_class.name not in mapping:
            new_labels.append(label)
        else:
            new_labels.append(label.clone(obj_class=mapping[label.obj_class.name]))

    new_ann = ann.clone(labels=new_labels)
    return (res_meta, res_meta, new_ann)