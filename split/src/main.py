from copy import copy, deepcopy
import math
import os
import random

import cv2
import globals as g
import imgaug.augmenters as iaa
import init_ui
import supervisely as sly
from supervisely.geometry.sliding_windows_fuzzy import (
    SlidingWindowBorderStrategy,
    SlidingWindowsFuzzy,
)
from tqdm import tqdm


def cache_images_info(api: sly.Api, project_id):
    for dataset_info in api.dataset.get_list(project_id):
        g.IMAGES_INFO.extend(api.image.get_list(dataset_info.id))


def refresh_progress_preview(api: sly.Api, task_id, progress: tqdm):
    fields = [
        {"field": "data.progressPreview", "payload": int(progress.n * 100 / progress.total)},
        {"field": "data.progressPreviewMessage", "payload": progress.desc},
        {"field": "data.progressPreviewCurrent", "payload": progress.n},
        {"field": "data.progressPreviewTotal", "payload": progress.total},
    ]
    api.task.set_fields(task_id, fields)


def check_sliding_sizes_by_image(img_info, state):
    if state["windowHeight"] > img_info.height:
        state["windowHeight"] = img_info.height

    if state["windowWidth"] > img_info.width:
        state["windowWidth"] = img_info.width

    return 0


def get_sliding_windows_sizes(image_info, state):
    if state["usePercents"] is True:
        w_height = math.ceil(image_info.height * state["windowHeightPercent"] / 100)
        overlap_y = math.ceil(w_height * state["overlapYPercent"] / 100)
        if state["useSquare"] is True:
            w_width = w_height
            overlap_x = overlap_y
        else:
            w_width = math.ceil(image_info.width * state["windowWidthPercent"] / 100)
            overlap_x = math.ceil(w_width * state["overlapXPercent"] / 100)
    else:
        w_height = state["windowHeightPx"]
        w_width = state["windowWidthPx"] if state["useSquare"] is False else w_height
        overlap_y = state["overlapYPx"]
        overlap_x = state["overlapXPx"] if state["useSquare"] is False else overlap_y

    state["windowHeight"] = w_height
    state["windowWidth"] = w_width
    state["overlapY"] = overlap_y
    state["overlapX"] = overlap_x


def _handle_error_and_exit(api: sly.Api, task_id: int, msg: str):
    sly.logger.warn(msg, exc_info=True)
    fields = [
        {"field": "data.videoUrl", "payload": None},
        {"field": "state.previewLoading", "payload": False},
    ]
    api.task.set_fields(task_id, fields)
    g.app.show_modal_window(msg, level="error", log_message=False)
    return


@g.app.callback("preview")
@sly.timeit
def preview(api: sly.Api, task_id, context, state, app_logger):
    if len(g.IMAGES_INFO) == 0:
        message = f"Project {g.PROJECT_INFO.name} has no images"
        description = "Please, check your project and try again."
        api.task.set_output_error(task_id, message, description)
        g.app.show_modal_window(message, level="error")
        g.app.stop()
        return
    fields = [
        {"field": "data.videoUrl", "payload": None},
        {"field": "state.previewLoading", "payload": True},
    ]
    api.task.set_fields(task_id, fields)

    image_info = random.choice(g.IMAGES_INFO)
    get_sliding_windows_sizes(image_info=image_info, state=state)
    check_sliding_sizes_by_image(img_info=image_info, state=state)

    try:
        slider = SlidingWindowsFuzzy(
            [state["windowHeight"], state["windowWidth"]],
            [state["overlapY"], state["overlapX"]],
            state["borderStrategy"],
        )
    except (ValueError, RuntimeError) as re:
        _handle_error_and_exit(api=api, task_id=task_id, msg=f"Wrong sliding window settings: {re}")
        return
    except Exception as e:
        _handle_error_and_exit(api=api, task_id=task_id, msg=f"Unexpected error: {repr(e)}")
        return

    img = api.image.download_np(image_info.id)

    ann_json = api.annotation.download(image_info.id).annotation
    ann = sly.Annotation.from_json(ann_json, g.PROJECT_META)

    if state["drawLabels"] is True:
        if state["cleanLabels"] is True:
            i = 0
            label_id_to_area = {}
            labels = []
            for label in ann.labels:
                label = label.clone(description=str(i))
                label_id_to_area[i] = label.area
                labels.append(label)
                i += 1
            ann = ann.clone(labels=labels)

    h, w = img.shape[:2]
    max_right = w - 1
    max_bottom = h - 1
    rectangles = []
    for window in slider.get(img.shape[:2]):
        rectangles.append(window)
        max_right = max(max_right, window.right)
        max_bottom = max(max_bottom, window.bottom)

    if max_right > w or max_bottom > h:
        sly.logger.debug(
            "Padding", extra={"h": h, "w": w, "max_right": max_right, "max_bottom": max_bottom}
        )
        aug = iaa.PadToFixedSize(width=max_right, height=max_bottom, position="right-bottom")
        img = aug(image=img)
        # sly.image.write(os.path.join(app.data_dir, "padded.jpg"), img)

    frame_img = img.copy()
    resize_aug = None
    if frame_img.shape[0] > g.MAX_VIDEO_HEIGHT:
        resize_aug = iaa.Resize({"height": g.MAX_VIDEO_HEIGHT, "width": "keep-aspect-ratio"})
        frame_img = resize_aug(image=frame_img.copy())
    height, width, channels = frame_img.shape

    video_path = os.path.join(g.app.data_dir, "preview.mp4")
    sly.fs.ensure_base_path(video_path)
    sly.fs.silent_remove(video_path)
    video = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"VP90"), state["fps"], (width, height)
    )
    report_every = max(5, math.ceil(len(rectangles) / 100))
    progress = tqdm(desc="Rendering frames", total=len(rectangles))
    refresh_progress_preview(api, task_id, progress)
    for i, rect in enumerate(rectangles):
        frame = img.copy()
        if state["drawLabels"] is True:
            crop_ann = ann.relative_crop(rect)
            temp_crop_img = frame[rect.top : rect.bottom + 1, rect.left : rect.right + 1].copy()
            if state["cleanLabels"] is True:
                filtered_labels = []
                for label in crop_ann.labels:
                    full_area = label_id_to_area[int(label.description)]
                    if full_area == 0:
                        continue
                    if label.area / full_area * 100 > state["cleanLabelsThreshold"]:
                        filtered_labels.append(label)
                crop_ann = crop_ann.clone(labels=filtered_labels)
            crop_ann.draw_pretty(temp_crop_img, thickness=3)
            frame[rect.top : rect.bottom + 1, rect.left : rect.right + 1] = temp_crop_img
        rect: sly.Rectangle
        rect.draw_contour(frame, [255, 0, 0], thickness=5)
        if resize_aug is not None:
            frame = resize_aug(image=frame)
        # sly.image.write(os.path.join(app.data_dir, f"{i:05d}.jpg"), frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)

        progress.update()
        if i % report_every == 0:
            refresh_progress_preview(api, task_id, progress)

    progress = tqdm(desc="Saving video file", total=1)
    progress.update()
    refresh_progress_preview(api, task_id, progress)
    video.release()

    progress = tqdm(desc="Uploading video", total=1)
    progress.update()
    refresh_progress_preview(api, task_id, progress)
    remote_video_path = os.path.join(f"/sliding-window/{task_id}", "preview.mp4")
    if api.file.exists(g.TEAM_ID, remote_video_path):
        api.file.remove(g.TEAM_ID, remote_video_path)
    file_info = api.file.upload(g.TEAM_ID, video_path, remote_video_path)

    fields = [
        {"field": "state.previewLoading", "payload": False},
        {"field": "data.videoUrl", "payload": file_info.storage_path},
    ]
    api.task.set_fields(task_id, fields)


def refresh_progress_split(api: sly.Api, task_id, progress: tqdm):
    fields = [
        {"field": "data.progress", "payload": int(progress.n * 100 / progress.total)},
        {"field": "data.progressCurrent", "payload": progress.n},
        {"field": "data.progressTotal", "payload": progress.total},
    ]
    api.task.set_fields(task_id, fields)


@g.app.callback("split")
@sly.timeit
def split(api: sly.Api, task_id, context, state, app_logger):
    if len(g.IMAGES_INFO) == 0:
        message = f"Project {g.PROJECT_INFO.name} has no images"
        description = "Please, check your project and try again."
        api.task.set_output_error(task_id, message, description)
        g.app.show_modal_window(message, level="error")
        g.app.stop()
        return
    image_info = random.choice(g.IMAGES_INFO)
    get_sliding_windows_sizes(image_info=image_info, state=state)
    # slider = SlidingWindowsFuzzy(
    #     [state["windowHeight"], state["windowWidth"]],
    #     [state["overlapY"], state["overlapX"]],
    #     state["borderStrategy"],
    # )

    dst_project = api.project.create(
        g.WORKSPACE_ID, state["resProjectName"], change_name_if_conflict=True
    )
    if dst_project.name != state["resProjectName"]:
        sly.logger.warn(
            "Project with name={!r} already exists. Project is saved with autogenerated name {!r}".format(
                state["resProjectName"], dst_project.name
            )
        )
    api.project.update_meta(dst_project.id, g.PROJECT_META.to_json())

    px = state["usePercents"] is False
    windowHeight = f'{state["windowHeightPx"]}px' if px else f'{state["windowHeightPercent"]}%'
    windowWidth = f'{state["windowWidthPx"]}px' if px else f'{state["windowWidthPercent"]}%'
    overlapY = f'{state["overlapYPx"]}px' if px else f'{state["overlapYPercent"]}%'
    overlapX = f'{state["overlapXPx"]}px' if px else f'{state["overlapXPercent"]}%'

    custom_data = {
        "inputProject": {"id": g.PROJECT_INFO.id, "name": g.PROJECT_INFO.name},
        "slidingWindow": {
            "windowHeight": windowHeight,
            "windowWidth": windowWidth,
            "overlapY": overlapY,
            "overlapX": overlapX,
            "borderStrategy": state["borderStrategy"],
        },
        "taskId": task_id,
    }
    sly.logger.info(f"Starting split with settings: {state}")
    api.project.update_custom_data(
        dst_project.id,
        data=custom_data,
    )
    dst_datasets = {}

    progress = tqdm(desc="Splitting images", total=len(g.IMAGES_INFO))

    state_backup = deepcopy(state)

    for image_info in g.IMAGES_INFO:
        get_sliding_windows_sizes(image_info=image_info, state=state)
        check_sliding_sizes_by_image(img_info=image_info, state=state)

        try:
            slider = SlidingWindowsFuzzy(
                [state["windowHeight"], state["windowWidth"]],
                [state["overlapY"], state["overlapX"]],
                state["borderStrategy"],
            )
        except (ValueError, RuntimeError) as re:
            _handle_error_and_exit(
                api=api, task_id=task_id, msg=f"Wrong sliding window settings: {re}"
            )
            return
        except Exception as e:
            _handle_error_and_exit(api=api, task_id=task_id, msg=f"Unexpected error: {repr(e)}")
            return

        if image_info.dataset_id not in dst_datasets:
            dataset_info = api.dataset.get_info_by_id(image_info.dataset_id)
            dst_datasets[image_info.dataset_id] = api.dataset.create(
                dst_project.id, dataset_info.name, dataset_info.description
            )
        dst_dataset = dst_datasets[image_info.dataset_id]

        img = api.image.download_np(image_info.id)
        ann_json = api.annotation.download(image_info.id).annotation
        ann = sly.Annotation.from_json(ann_json, g.PROJECT_META)

        if state["cleanLabels"] is True:
            # create temporary annotation (set index as description for each label)
            i = 0
            label_id_to_area = {}
            labels = []
            for label in ann.labels:
                label = label.clone(description=str(i))
                label_id_to_area[i] = label.area
                labels.append(label)
                i += 1
            new_ann = ann.clone(labels=labels)

        crop_names = []
        crop_images = []
        crop_anns = []

        for window_index, window in enumerate(slider.get(img.shape[:2])):
            safe_base_name = sly.fs.get_file_name(image_info.name).replace("___", "__")
            if window_index == 0:
                crop_name = "{}___{:04d}_{}_{}_dims_{}x{}{}".format(
                    safe_base_name,
                    window_index,
                    window.top,
                    window.left,
                    img.shape[0],
                    img.shape[1],
                    sly.fs.get_file_ext(image_info.name),
                )
            else:
                crop_name = "{}___{:04d}_{}_{}{}".format(
                    safe_base_name,
                    window_index,
                    window.top,
                    window.left,
                    sly.fs.get_file_ext(image_info.name),
                )

            crop_ann = ann.relative_crop(window)
            if state["cleanLabels"] is True:
                # will use temporary annotation to match labels areas with same labels in full image
                temp_crop = new_ann.relative_crop(window)

                filtered_labels = []
                for label, temp_label in zip(
                    crop_ann.labels, temp_crop.labels
                ):  # labels are in the same order
                    full_area = label_id_to_area[int(temp_label.description)]
                    if full_area == 0:
                        continue
                    if label.area / full_area * 100 > state["cleanLabelsThreshold"]:
                        filtered_labels.append(label)
                crop_ann = crop_ann.clone(labels=filtered_labels)

            if state["borderStrategy"] == str(SlidingWindowBorderStrategy.ADD_PADDING):
                crop_image = sly.image.crop_with_padding(img, window)
            else:
                crop_image = sly.image.crop(img, window)
            if state["resizeWindow"] is True:
                resize_aug = iaa.Resize(
                    {"height": state["resizeValue"], "width": "keep-aspect-ratio"}
                )
                resized_image = resize_aug(image=crop_image.copy())
                try:
                    resized_ann = crop_ann.resize(resized_image.shape[:2])
                    crop_anns.append(resized_ann)
                    crop_images.append(resized_image)
                except Exception as e:
                    sly.logger.warn(f"Can not resize {image_info.name} image and annotations.")
                    crop_images.append(crop_image)
                    crop_anns.append(crop_ann)
            else:
                crop_images.append(crop_image)
                crop_anns.append(crop_ann)

            crop_names.append(crop_name)

        dst_image_infos = api.image.upload_nps(dst_dataset.id, crop_names, crop_images)
        dst_image_ids = [dst_img_info.id for dst_img_info in dst_image_infos]
        api.annotation.upload_anns(dst_image_ids, crop_anns)

        progress.update(1)
        refresh_progress_split(api, task_id, progress)

        state = deepcopy(state_backup)

    res_project = api.project.get_info_by_id(dst_project.id)
    fields = [
        {"field": "data.started", "payload": False},
        {"field": "data.resProjectId", "payload": res_project.id},
        {"field": "data.resProjectName", "payload": res_project.name},
        {
            "field": "data.resProjectPreviewUrl",
            "payload": api.image.preview_url(res_project.reference_image_url, 100, 100),
        },
    ]
    api.task.set_fields(task_id, fields)
    api.task.set_output_project(task_id, res_project.id, res_project.name)
    g.app.stop()


def main():
    data = {}
    state = {}

    init_ui.init_input_project(g.app.public_api, data, g.PROJECT_INFO)
    init_ui.init_settings(state)
    init_ui.init_res_project(data, state, g.PROJECT_INFO)

    data["videoUrl"] = None

    data["progress"] = 0
    data["progressCurrent"] = 0
    data["progressTotal"] = 0

    state["previewLoading"] = False
    data["progressPreview"] = 0
    data["progressPreviewMessage"] = "Rendering frames"
    data["progressPreviewCurrent"] = 0
    data["progressPreviewTotal"] = 0

    cache_images_info(g.app.public_api, g.PROJECT_ID)
    g.app.run(data=data, state=state)


# https://github.com/supervisely/supervisely/tree/master/plugins/python/src/examples/001_image_splitter
if __name__ == "__main__":
    sly.main_wrapper("main", main)
