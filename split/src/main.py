import os
import random
import cv2
import supervisely_lib as sly
import math
import imgaug.augmenters as iaa
from supervisely_lib.geometry.sliding_windows_fuzzy import SlidingWindowsFuzzy, SlidingWindowBorderStrategy

import init_ui

app: sly.AppService = sly.AppService()

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

project_info = app.public_api.project.get_info_by_id(project_id)
if project_info is None:
    raise RuntimeError(f"Project id={project_id} not found")

meta = sly.ProjectMeta.from_json(app.public_api.project.get_meta(project_id))
if len(meta.obj_classes) == 0:
    raise ValueError("Project should have at least one class")

images_info = []

MAX_VIDEO_HEIGHT = 800  # in pixels


def cache_images_info(api: sly.Api, project_id):
    global images_info
    for dataset_info in api.dataset.get_list(project_id):
        images_info.extend(api.image.get_list(dataset_info.id))


def refresh_progress_preview(api: sly.Api, task_id, progress: sly.Progress):
    fields = [
        {"field": "data.progressPreview", "payload": int(progress.current * 100 / progress.total)},
        {"field": "data.progressPreviewMessage", "payload": progress.message},
        {"field": "data.progressPreviewCurrent", "payload": progress.current},
        {"field": "data.progressPreviewTotal", "payload": progress.total},
    ]
    api.task.set_fields(task_id, fields)


@app.callback("preview")
@sly.timeit
def preview(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.videoUrl", "payload": None},
        {"field": "state.previewLoading", "payload": True},
    ]
    api.task.set_fields(task_id, fields)

    slider = SlidingWindowsFuzzy([state["windowHeight"], state["windowWidth"]],
                                 [state["overlapY"], state["overlapX"]],
                                 state["borderStrategy"])
    image_info = random.choice(images_info)
    img = api.image.download_np(image_info.id)

    ann_json = api.annotation.download(image_info.id).annotation
    ann = sly.Annotation.from_json(ann_json, meta)

    if state["drawLabels"] is True:
        ann.draw_pretty(img, thickness=3)

    h, w = img.shape[:2]
    max_right = w - 1
    max_bottom = h - 1
    rectangles = []
    for window in slider.get(img.shape[:2]):
        rectangles.append(window)
        max_right = max(max_right, window.right)
        max_bottom = max(max_bottom, window.bottom)

    if max_right > w or max_bottom > h:
        sly.logger.debug("Padding", extra={"h": h, "w": w, "max_right": max_right, "max_bottom": max_bottom})
        aug = iaa.PadToFixedSize(width=max_right, height=max_bottom, position='right-bottom')
        img = aug(image=img)
        #sly.image.write(os.path.join(app.data_dir, "padded.jpg"), img)

    frame_img = img.copy()
    resize_aug = None
    if frame_img.shape[0] > MAX_VIDEO_HEIGHT:
        resize_aug = iaa.Resize({"height": MAX_VIDEO_HEIGHT, "width": "keep-aspect-ratio"})
        frame_img = resize_aug(image=frame_img.copy())
    height, width, channels = frame_img.shape

    video_path = os.path.join(app.data_dir, "preview.mp4")
    sly.fs.ensure_base_path(video_path)
    sly.fs.silent_remove(video_path)
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'VP90'), state["fps"], (width, height))
    report_every = max(5, math.ceil(len(rectangles) / 100))
    progress = sly.Progress("Rendering frames", len(rectangles))
    refresh_progress_preview(api, task_id, progress)
    for i, rect in enumerate(rectangles):
        frame = img.copy()
        rect: sly.Rectangle
        rect.draw_contour(frame, [255, 0, 0], thickness=5)
        if resize_aug is not None:
            frame = resize_aug(image=frame)
        #sly.image.write(os.path.join(app.data_dir, f"{i:05d}.jpg"), frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)

        progress.iter_done_report()
        if i % report_every == 0:
            refresh_progress_preview(api, task_id, progress)

    progress = sly.Progress("Saving video file", 1)
    progress.iter_done_report()
    refresh_progress_preview(api, task_id, progress)
    video.release()

    progress = sly.Progress("Uploading video", 1)
    progress.iter_done_report()
    refresh_progress_preview(api, task_id, progress)
    remote_video_path = os.path.join(f"/sliding-window/{task_id}", "preview.mp4")
    if api.file.exists(team_id, remote_video_path):
        api.file.remove(team_id, remote_video_path)
    file_info = api.file.upload(team_id, video_path, remote_video_path)

    fields = [
        {"field": "state.previewLoading", "payload": False},
        {"field": "data.videoUrl", "payload": file_info.full_storage_url},
    ]
    api.task.set_fields(task_id, fields)


def refresh_progress_split(api: sly.Api, task_id, progress: sly.Progress):
    fields = [
        {"field": "data.progress", "payload": int(progress.current * 100 / progress.total)},
        {"field": "data.progressCurrent", "payload": progress.current},
        {"field": "data.progressTotal", "payload": progress.total},
    ]
    api.task.set_fields(task_id, fields)


@app.callback("split")
@sly.timeit
def split(api: sly.Api, task_id, context, state, app_logger):
    slider = SlidingWindowsFuzzy([state["windowHeight"], state["windowWidth"]],
                                 [state["overlapY"], state["overlapX"]],
                                 state["borderStrategy"])

    dst_project = api.project.create(workspace_id, state["resProjectName"], change_name_if_conflict=True)
    if dst_project.name != state["resProjectName"]:
        sly.logger.warn("Project with name={!r} already exists. Project is saved with autogenerated name {!r}"
                        .format(state["resProjectName"], dst_project.name))
    api.project.update_meta(dst_project.id, meta.to_json())
    api.project.update_custom_data(dst_project.id, {
        "inputProject": {
            "id": project_info.id,
            "name": project_info.name
        },
        "slidingWindow": {
            "windowHeight": state["windowHeight"],
            "windowWidth": state["windowWidth"],
            "overlapY": state["overlapY"],
            "overlapX": state["overlapY"],
            "borderStrategy": state["borderStrategy"]
        }
    })
    dst_datasets = {}

    progress = sly.Progress("SW split", len(images_info))
    for image_info in images_info:
        if image_info.dataset_id not in dst_datasets:
            dataset_info = api.dataset.get_info_by_id(image_info.dataset_id)
            dst_datasets[image_info.dataset_id] = api.dataset.create(dst_project.id, dataset_info.name, dataset_info.description)
        dst_dataset = dst_datasets[image_info.dataset_id]

        img = api.image.download_np(image_info.id)
        ann_json = api.annotation.download(image_info.id).annotation
        ann = sly.Annotation.from_json(ann_json, meta)

        crop_names = []
        crop_images = []
        crop_anns = []

        for window_index, window in enumerate(slider.get(img.shape[:2])):
            crop_name = "{}___{:04d}_{}_{}{}".format(sly.fs.get_file_name(image_info.name),
                                                     window_index,
                                                     window.top,
                                                     window.left,
                                                     sly.fs.get_file_ext(image_info.name))
            crop_names.append(crop_name)

            crop_ann = ann.relative_crop(window)
            crop_anns.append(crop_ann)

            if state["borderStrategy"] == str(SlidingWindowBorderStrategy.ADD_PADDING):
                crop_image = sly.image.crop_with_padding(img, window)
            else:
                crop_image = sly.image.crop(img, window)
            crop_images.append(crop_image)

        dst_image_infos = api.image.upload_nps(dst_dataset.id, crop_names, crop_images)
        dst_image_ids = [dst_img_info.id for dst_img_info in dst_image_infos]
        api.annotation.upload_anns(dst_image_ids, crop_anns)

        progress.iter_done_report()
        if progress.need_report():
            refresh_progress_split(api, task_id, progress)

    res_project = api.project.get_info_by_id(dst_project.id)
    fields = [
        {"field": "data.started", "payload": False},
        {"field": "data.resProjectId", "payload": res_project.id},
        {"field": "data.resProjectName", "payload": res_project.name},
        {"field": "data.resProjectPreviewUrl",
         "payload": api.image.preview_url(res_project.reference_image_url, 100, 100)},
    ]
    api.task.set_fields(task_id, fields)
    api.task.set_output_project(task_id, res_project.id, res_project.name)
    app.stop()


def main():
    data = {}
    state = {}

    init_ui.init_input_project(app.public_api, data, project_info)
    init_ui.init_settings(state)
    init_ui.init_res_project(data, state, project_info)

    data["videoUrl"] = None

    data["progress"] = 0
    data["progressCurrent"] = 0
    data["progressTotal"] = 0

    state["previewLoading"] = False
    data["progressPreview"] = 0
    data["progressPreviewMessage"] = "Rendering frames"
    data["progressPreviewCurrent"] = 0
    data["progressPreviewTotal"] = 0

    cache_images_info(app.public_api, project_id)
    app.run(data=data, state=state)


# https://github.com/supervisely/supervisely/tree/master/plugins/python/src/examples/001_image_splitter
if __name__ == "__main__":
    sly.main_wrapper("main", main)
