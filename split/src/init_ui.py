import os

import supervisely as sly


def init_input_project(api: sly.Api, data: dict, project_info):
    data["projectId"] = project_info.id
    data["projectName"] = project_info.name
    data["projectPreviewUrl"] = api.image.preview_url(project_info.reference_image_url, 100, 100)
    data["projectItemsCount"] = project_info.items_count


def init_settings(state):
    state["windowHeight"] = 256
    state["windowWidth"] = 256
    state["overlapY"] = 32
    state["overlapX"] = 32

    state["usePercents"] = False
    state["useSquare"] = False
    state["windowHeightPx"] = 256
    state["windowWidthPx"] = 256
    state["overlapYPx"] = 32
    state["overlapXPx"] = 32

    state["windowHeightPercent"] = 20
    state["windowWidthPercent"] = 20
    state["overlapYPercent"] = 2
    state["overlapXPercent"] = 2

    state["resizeWindow"] = False
    state["resizeValue"] = 0

    state["borderStrategy"] = "shift_window"  # "add_padding"
    state["fps"] = 4
    state["drawLabels"] = True


def init_res_project(data, state, project_info):
    data["resProjectId"] = None
    state["resProjectName"] = f"{project_info.name}-sw-split"
    data["resProjectName"] = None
    data["resProjectPreviewUrl"] = None
    data["started"] = False
