import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import supervisely_lib as sly
from supervisely.app.v1.app_service import AppService

root_source_path = str(Path(sys.argv[0]).parents[2])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

# only for debug
load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("split/debug.env")

app: AppService = AppService()

TEAM_ID = sly.env.team_id()
WORKSPACE_ID = sly.env.workspace_id()
PROJECT_ID = sly.env.project_id()

PROJECT_INFO = app.public_api.project.get_info_by_id(PROJECT_ID)
if PROJECT_INFO is None:
    raise RuntimeError(f"Project id={PROJECT_ID} not found")

PROJECT_META = sly.ProjectMeta.from_json(app.public_api.project.get_meta(PROJECT_ID))
# if len(meta.obj_classes) == 0:
#     raise ValueError("Project should have at least one class")

IMAGES_INFO = []

MAX_VIDEO_HEIGHT = 800  # in pixels
