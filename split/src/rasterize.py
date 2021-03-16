import numpy as np
import supervisely_lib as sly


def need_convert(geometry_type) -> bool:
    if geometry_type in [sly.Polygon, sly.Rectangle, sly.Bitmap, sly.AnyGeometry]:
        return True
    return False


def allow_render_for_any_shape(lbl: sly.Label):
    if lbl.obj_class.geometry_type == sly.AnyGeometry and need_convert(type(lbl.geometry)) is False:
        return False
    return True


def convert_to_nonoverlapping(meta: sly.ProjectMeta, ann: sly.Annotation) -> (sly.ProjectMeta, sly.Annotation):
    common_img = np.zeros(ann.img_size, np.int32)  # size is (h, w)
    for idx, lbl in enumerate(ann.labels, start=1):
        if need_convert(lbl.obj_class.geometry_type):
            if allow_render_for_any_shape(lbl) is True:
                lbl.draw(common_img, color=idx)
            else:
                sly.logger.warn(
                    "Object of class {!r} (shape: {!r}) has non spatial shape {!r}. It will not be rendered."
                        .format(lbl.obj_class.name,
                                lbl.obj_class.geometry_type.geometry_name(),
                                lbl.geometry.geometry_name()))

    new_classes = sly.ObjClassCollection()
    new_labels = []
    for idx, lbl in enumerate(ann.labels, start=1):
        if not need_convert(lbl.obj_class.geometry_type):
            new_labels.append(lbl.clone())
        else:
            if allow_render_for_any_shape(lbl) is False:
                continue
            # @TODO: get part of the common_img for speedup
            mask = common_img == idx
            if np.any(mask):  # figure may be entirely covered by others
                g = lbl.geometry
                new_bmp = sly.Bitmap(data=mask)
                if new_classes.get(lbl.obj_class.name) is None:
                    new_classes = new_classes.add(lbl.obj_class.clone(geometry_type=sly.Bitmap))

                new_lbl = lbl.clone(geometry=new_bmp, obj_class=new_classes.get(lbl.obj_class.name))
                new_labels.append(new_lbl)

    new_meta = meta.clone(obj_classes=new_classes)
    new_ann = ann.clone(labels=new_labels)
    return (new_meta, new_ann)
