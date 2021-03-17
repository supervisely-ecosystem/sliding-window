import os
import random
import cv2
import supervisely_lib as sly
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

# CNT_GRID_COLUMNS = 1
# empty_gallery = {
#     "content": {
#         "projectMeta": sly.ProjectMeta().to_json(),
#         "annotations": {},
#         "layout": [[] for i in range(CNT_GRID_COLUMNS)]
#     },
#     "previewOptions": {
#         "enableZoom": True,
#         "resizeOnZoom": True
#     },
#     "options": {
#         "enableZoom": False,
#         "syncViews": False,
#         "showPreview": True,
#         "selectable": False,
#         "opacity": 0.5
#     }
# }


def cache_images_info(api: sly.Api, project_id):
    global images_info
    for dataset_info in api.dataset.get_list(project_id):
        images_info.extend(api.image.get_list(dataset_info.id))


@app.callback("split")
@sly.timeit
def generate(api: sly.Api, task_id, context, state, app_logger):
    pass


@app.callback("preview")
@sly.timeit
def preview(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.videoUrl", "payload": None},
    ]
    api.task.set_fields(task_id, fields)

    slider = SlidingWindowsFuzzy([state["windowHeight"], state["windowWidth"]],
                                 [state["overlapY"], state["overlapX"]],
                                 state["borderStrategy"])
    image_info = random.choice(images_info)
    img = api.image.download_np(image_info.id)

    ann_json = api.annotation.download(image_info.id).annotation
    ann = sly.Annotation.from_json(ann_json, meta)

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
    sly.fs.silent_remove(video_path)
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'avc1'), 2, (width, height))
    for i, rect in enumerate(rectangles):
        frame = img.copy()
        rect: sly.Rectangle
        rect.draw_contour(frame, [255, 0, 0], thickness=5)
        if resize_aug is not None:
            frame = resize_aug(image=frame)
        #sly.image.write(os.path.join(app.data_dir, f"{i:05d}.jpg"), frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
    video.release()

    remote_video_path = os.path.join(f"/sliding-window/{task_id}", "preview.mp4")
    file_info = None
    if api.file.exists(team_id, remote_video_path):
        api.file.remove(team_id, remote_video_path)
    file_info = api.file.upload(team_id, video_path, remote_video_path)

    #print(file_info.full_storage_url)
    fields = [
        {"field": "state.previewLoading", "payload": False},
        {"field": "data.videoUrl", "payload": file_info.full_storage_url},
    ]
    api.task.set_fields(task_id, fields)


def main():
    data = {}
    state = {}

    init_ui.init_input_project(app.public_api, data, project_info)
    init_ui.init_settings(state)
    data["videoUrl"] = None
    cache_images_info(app.public_api, project_id)

    app.run(data=data, state=state)


# https://github.com/supervisely/supervisely/tree/master/plugins/python/src/examples/001_image_splitter
if __name__ == "__main__":
    sly.main_wrapper("main", main)
