from collections import defaultdict

import globals as g
import numpy as np
import supervisely as sly
from supervisely.geometry.sliding_windows_fuzzy import SlidingWindowBorderStrategy


class Regexps:
    filename_re = r"\w+(?=\___)"
    settings_re = r"(?<=\___)\w+"
    dim_re = r"(?<=_dims_)\d+x\d+(?=\.)"

    @staticmethod
    def extract_by_regexp(filename, regexp):
        import re

        title_search = re.search(pattern=regexp, string=filename)
        if title_search:
            return title_search.group(0)

    @staticmethod
    def get_ext(filename):
        return filename[filename.rindex(".") :]

    @staticmethod
    def get_orig_dimensions(filename, regexp):
        """Dimensions are stored in filename as _dims_<height>x<width> and placed before file extension in the 0000 part"""
        import re

        dim_search = re.search(pattern=regexp, string=filename)
        if dim_search:
            dims_str = dim_search.group(0)
            height, width = map(int, dims_str.split("x"))
            return height, width
        return None


@g.app.callback("merge")
@sly.timeit
def merge(api: sly.Api, task_id, context, state, app_logger):
    dst_project = api.project.create(
        g.WORKSPACE_ID, g.DST_PROJECT_NAME, change_name_if_conflict=True
    )
    api.project.update_meta(dst_project.id, g.PROJECT_META.to_json())
    api.project.update_custom_data(
        dst_project.id, {"input_project": {"id": g.SRC_PROJECT.id, "name": g.SRC_PROJECT.name}}
    )

    # Get sliding window settings from source project metadata
    src_custom_data = api.project.get_info_by_id(g.SRC_PROJECT.id).custom_data
    sliding_window_settings = src_custom_data.get("slidingWindow", {})

    # Extract strategy
    border_strategy = sliding_window_settings.get("borderStrategy", "none")

    # Extract overlap values
    overlap_x_str = sliding_window_settings.get("overlapX", "0px")
    overlap_y_str = sliding_window_settings.get("overlapY", "0px")

    # Parse overlap values (remove 'px' or '%' suffix)
    overlap_x = int(overlap_x_str.replace("px", "").replace("%", ""))
    overlap_y = int(overlap_y_str.replace("px", "").replace("%", ""))

    sly.logger.info(f"Sliding window overlap: X={overlap_x}px, Y={overlap_y}px")

    progress = sly.Progress("Merging images", api.project.get_images_count(g.SRC_PROJECT.id))
    for src_dataset in api.dataset.get_list(g.SRC_PROJECT.id):
        dst_dataset = api.dataset.create(dst_project.id, src_dataset.name)
        images = api.image.get_list(src_dataset.id)
        image_ids = [image_info.id for image_info in images]
        # image_names = [image_info.name for image_info in images]

        ann_infos = api.annotation.download_batch(src_dataset.id, image_ids)
        anns = [
            sly.Annotation.from_json(ann_info.annotation, g.PROJECT_META) for ann_info in ann_infos
        ]

        parts_ids = defaultdict(list)
        parts_top_left = defaultdict(list)
        parts_anns = defaultdict(list)
        original_dims = defaultdict(dict)

        max_height = defaultdict(int)
        max_width = defaultdict(int)
        for image_info, ann in zip(images, anns):
            # sly.logger.info(f'{image_info.name=}')
            if image_info.name.count("___") != 1:
                raise RuntimeError(
                    "Incorrect images names. Should be: "
                    "<image name>___<window index>_<window top coordinate>_<window left coordinate>.<image extension> "
                    "or <image name>___<window index>_<window top coordinate>_<window left coordinate>_<orig image dimensions>.<image extension> "
                    "Use Sliding window split app first to correctly split images."
                )

            real_name = image_info.name.split("___")[0]
            # real_name = Regexps.extract_by_regexp(image_info.name, Regexps.filename_re)
            ext = Regexps.get_ext(image_info.name)
            settings = Regexps.extract_by_regexp(image_info.name, Regexps.settings_re)
            original_dims_info = Regexps.get_orig_dimensions(image_info.name, Regexps.dim_re)

            if settings is None:
                raise RuntimeError(
                    "Incorrect images names. Should be: "
                    "<image name>___<window index>_<window top coordinate>_<window left coordinate>.<image extension> "
                    "or <image name>___<window index>_<window top coordinate>_<window left coordinate>_<orig image dimensions>.<image extension> "
                    "Use Sliding window split app first to correctly split images."
                )

            # window_index = int(settings.split("_")[0])
            window_top = int(settings.split("_")[1])
            window_left = int(settings.split("_")[2])

            original_name = "{}{}".format(real_name, ext)

            max_height[original_name] = max(max_height[original_name], window_top + ann.img_size[0])
            max_width[original_name] = max(max_width[original_name], window_left + ann.img_size[1])
            parts_ids[original_name].append(image_info.id)
            parts_top_left[original_name].append((window_top, window_left))
            parts_anns[original_name].append(ann)
            if original_dims_info is not None:
                original_dims[original_name]["height"] = original_dims_info[0]
                original_dims[original_name]["width"] = original_dims_info[1]

        for original_name in parts_ids.keys():
            images = api.image.download_nps(src_dataset.id, parts_ids[original_name])
            height = max_height[original_name]
            width = max_width[original_name]
            channels = images[0].shape[2]
            final_image = np.zeros((height, width, channels), dtype=np.uint8)
            final_ann = sly.Annotation(final_image.shape[:2])

            # Group windows by position for easier overlap detection
            windows_info = []
            for image_part, ann, (top, left) in zip(
                images, parts_anns[original_name], parts_top_left[original_name]
            ):
                windows_info.append(
                    {
                        "image": image_part,
                        "ann": ann,
                        "top": top,
                        "left": left,
                        "bottom": top + image_part.shape[0],
                        "right": left + image_part.shape[1],
                        "height": image_part.shape[0],
                        "width": image_part.shape[1],
                    }
                )

            # Sort windows by position (top-to-bottom, left-to-right)
            windows_info.sort(key=lambda x: (x["top"], x["left"]))

            # Calculate grid width (number of windows per row)
            # Count how many windows have the same top coordinate as the first window
            if len(windows_info) > 0:
                first_top = windows_info[0]["top"]
                grid_width = sum(1 for w in windows_info if w["top"] == first_top)
            else:
                grid_width = 0

            # Process each window
            for idx, window in enumerate(windows_info):
                top, left = window["top"], window["left"]
                window_h, window_w = window["height"], window["width"]

                # Calculate the region of this window that should contain unique annotations
                # Each window owns half of the overlap with its neighbors

                crop_left = 0  # Start from beginning by default
                crop_top = 0  # Start from beginning by default
                crop_right = window_w  # Full width by default
                crop_bottom = window_h  # Full height by default

                # Left neighbor (idx - 1, but only if same row)
                left_idx = idx - 1
                if left_idx >= 0 and windows_info[left_idx]["top"] == top:
                    other = windows_info[left_idx]
                    other_right = other["left"] + other["width"]
                    if other_right > left:
                        # There is actual overlap
                        actual_overlap = other_right - left
                        # This window takes the right half of overlap (rounded up for odd numbers)
                        crop_left = (actual_overlap + 1) // 2
                    # else: no overlap (gap between windows), don't crop

                # Right neighbor (idx + 1, but only if same row)
                right_idx = idx + 1
                if right_idx < len(windows_info) and windows_info[right_idx]["top"] == top:
                    other = windows_info[right_idx]
                    this_right = left + window_w
                    if this_right > other["left"]:
                        # There is actual overlap
                        actual_overlap = this_right - other["left"]
                        # This window takes the left half of overlap (rounded down for odd numbers)
                        # Subtract 1 because Rectangle uses inclusive bounds
                        crop_right = window_w - ((actual_overlap + 1) // 2) - 1
                    # else: no overlap (gap between windows), don't crop

                # Top neighbor (idx - grid_width)
                top_idx = idx - grid_width
                if top_idx >= 0:
                    other = windows_info[top_idx]
                    # Verify it's actually above (same column)
                    if other["left"] == left:
                        other_bottom = other["top"] + other["height"]
                        if other_bottom > top:
                            # There is actual overlap
                            actual_overlap = other_bottom - top
                            # This window takes the bottom half of overlap (rounded up for odd numbers)
                            crop_top = (actual_overlap + 1) // 2
                        # else: no overlap (gap between windows), don't crop

                # Bottom neighbor (idx + grid_width)
                bottom_idx = idx + grid_width
                if bottom_idx < len(windows_info):
                    other = windows_info[bottom_idx]
                    # Verify it's actually below (same column)
                    if other["left"] == left:
                        this_bottom = top + window_h
                        if this_bottom > other["top"]:
                            # There is actual overlap
                            actual_overlap = this_bottom - other["top"]
                            # This window takes the top half of overlap (rounded down for odd numbers)
                            # Subtract 1 because Rectangle uses inclusive bounds
                            crop_bottom = window_h - ((actual_overlap + 1) // 2) - 1
                        # else: no overlap (gap between windows), don't crop

                # Translate and crop labels
                ann = window["ann"]

                def _translate_and_crop_label(label):
                    # Translate to global coordinates
                    translated = label.translate(top, left)

                    # Check if this window has a neighbor on the right or bottom
                    has_right_neighbor = (
                        right_idx < len(windows_info) and windows_info[right_idx]["top"] == top
                    )
                    has_bottom_neighbor = (
                        bottom_idx < len(windows_info) and windows_info[bottom_idx]["left"] == left
                    )

                    # Check if there's actual cropping on right/bottom (overlap scenario)
                    has_right_crop = crop_right < window_w
                    has_bottom_crop = crop_bottom < window_h

                    # During split, annotations lose 1 pixel on right/bottom edges
                    # We extend by 1 pixel ONLY if there's a neighbor BUT NO overlap cropping
                    geom = translated.geometry

                    # Extend geometry bounds to compensate for split crop loss
                    # BUT only if we're NOT already cropping due to overlap
                    if (has_right_neighbor and not has_right_crop) or (
                        has_bottom_neighbor and not has_bottom_crop
                    ):
                        bbox = geom.to_bbox()
                        extend_right = 1 if (has_right_neighbor and not has_right_crop) else 0
                        extend_bottom = 1 if (has_bottom_neighbor and not has_bottom_crop) else 0

                        new_right = bbox.right + extend_right
                        new_bottom = bbox.bottom + extend_bottom

                        # Create extended bbox
                        extended_bbox = sly.Rectangle(
                            top=bbox.top, left=bbox.left, bottom=new_bottom, right=new_right
                        )

                        # For simple geometries, extend them
                        if isinstance(geom, sly.Rectangle):
                            geom = extended_bbox
                        # For other geometries, we can't easily extend, so keep as is
                        translated = translated.clone(geometry=geom)

                    # If no cropping needed, return as is
                    if (
                        crop_left == 0
                        and crop_top == 0
                        and crop_right == window_w
                        and crop_bottom == window_h
                    ):
                        return [translated]

                    # Calculate crop rectangle in global coordinates (inclusive bounds)
                    crop_rect = sly.Rectangle(
                        top=top + crop_top,
                        left=left + crop_left,
                        bottom=top + crop_bottom,
                        right=left + crop_right,
                    )

                    # Crop the geometry
                    try:
                        geom = translated.geometry
                        cropped_geom = geom.crop(crop_rect)

                        if cropped_geom is None or (
                            isinstance(cropped_geom, list) and len(cropped_geom) == 0
                        ):
                            return []

                        # crop() can return a list of geometries
                        if isinstance(cropped_geom, list):
                            # Return labels for all pieces
                            return [
                                translated.clone(geometry=g)
                                for g in cropped_geom
                                if hasattr(g, "area") and g.area > 0
                            ]
                        else:
                            # Single geometry
                            if hasattr(cropped_geom, "area") and cropped_geom.area > 0:
                                return [translated.clone(geometry=cropped_geom)]
                            return []
                    except:
                        # If cropping fails, return original
                        return [translated]

                cropped_ann = ann.transform_labels(
                    _translate_and_crop_label, new_size=final_image.shape[:2]
                )

                # Add labels and tags
                final_ann = final_ann.add_labels(cropped_ann.labels)
                final_ann = final_ann.clone(
                    img_tags=final_ann.img_tags.merge_without_duplicates(cropped_ann.img_tags)
                )

                # Add image part
                final_image[top : top + window_h, left : left + window_w, :] = window["image"]

            # Adjust final image and annotation size if original dimensions are smaller (due to padding)
            if (
                border_strategy == str(SlidingWindowBorderStrategy.ADD_PADDING)
                and original_name in original_dims
            ):
                orig_height = original_dims[original_name].get("height", height)
                orig_width = original_dims[original_name].get("width", width)
                if orig_height < height or orig_width < width:
                    final_image = final_image[0:orig_height, 0:orig_width, :]
                    final_ann = final_ann.clone(img_size=(orig_height, orig_width))

            merged_image_info = api.image.upload_np(dst_dataset.id, original_name, final_image)
            api.annotation.upload_ann(merged_image_info.id, final_ann)
            progress.iters_done_report(len(images))

    api.task.set_output_project(task_id, dst_project.id, dst_project.name)
    g.app.stop()


def main():
    g.app.run(initial_events=[{"command": "merge"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main, log_for_agent=False)
