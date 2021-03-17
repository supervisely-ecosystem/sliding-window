import os
import random
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

MAX_VIDEO_HEIGHT = 600  # in pixels

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
        sly.image.write(os.path.join(app.data_dir, "padded.jpg"), img)


def main():
    data = {}
    state = {}

    init_ui.init_input_project(app.public_api, data, project_info)
    init_ui.init_settings(state)
    cache_images_info(app.public_api, project_id)

    app.run(data=data, state=state)

# https://github.com/supervisely/supervisely/tree/master/plugins/python/src/examples/001_image_splitter
if __name__ == "__main__":
    sly.main_wrapper("main", main)
