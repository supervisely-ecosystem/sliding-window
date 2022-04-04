import os
import numpy as np
from collections import defaultdict
import supervisely_lib as sly



class Regexps:
    filename_re = r"\w+(?=\___)"
    settings_re = r"(?<=\___)\w+"

    @staticmethod
    def extract_by_regexp(filename, regexp):
        import re
        title_search = re.search(pattern=regexp, string=filename)
        if title_search:
            return title_search.group(0)

    @staticmethod
    def get_ext(filename):
        return filename[filename.rindex('.'):]



app: sly.AppService = sly.AppService()

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

dst_project_name = os.environ['modal.state.resultProjectName']

src_project = app.public_api.project.get_info_by_id(project_id)
if src_project is None:
    raise RuntimeError(f"Project id={project_id} not found")

meta = sly.ProjectMeta.from_json(app.public_api.project.get_meta(project_id))
# if len(meta.obj_classes) == 0:
#     raise ValueError("Project should have at least one class")


@app.callback("merge")
@sly.timeit
def merge(api: sly.Api, task_id, context, state, app_logger):
    dst_project = api.project.create(workspace_id, dst_project_name, change_name_if_conflict=True)
    api.project.update_meta(dst_project.id, meta.to_json())
    api.project.update_custom_data(dst_project.id, {
        "input_project": {
            "id": src_project.id,
            "name": src_project.name
        }
    })

    progress = sly.Progress("Merging images", api.project.get_images_count(src_project.id))
    for src_dataset in api.dataset.get_list(src_project.id):
        dst_dataset = api.dataset.create(dst_project.id, src_dataset.name)
        images = api.image.get_list(src_dataset.id)
        image_ids = [image_info.id for image_info in images]
        #image_names = [image_info.name for image_info in images]

        ann_infos = api.annotation.download_batch(src_dataset.id, image_ids)
        anns = [sly.Annotation.from_json(ann_info.annotation, meta) for ann_info in ann_infos]

        parts_ids = defaultdict(list)
        parts_top_left = defaultdict(list)
        parts_anns = defaultdict(list)

        max_height = defaultdict(int)
        max_width = defaultdict(int)
        for image_info, ann in zip(images, anns):
            sly.logger.info(image_info.name)
            
            real_name = Regexps.extract_by_regexp(image_info.name, Regexps.filename_re)
            ext = Regexps.get_ext(image_info.name)
            settings = Regexps.extract_by_regexp(image_info.name, Regexps.settings_re)

            window_index = int(settings.split("_")[0])
            window_top = int(settings.split("_")[1])
            window_left = int(settings.split("_")[2])

            original_name = "{}{}".format(real_name, ext)

            max_height[original_name] = max(max_height[original_name], window_top + ann.img_size[0])
            max_width[original_name] = max(max_width[original_name], window_left + ann.img_size[1])
            parts_ids[original_name].append(image_info.id)
            parts_top_left[original_name].append((window_top, window_left))
            parts_anns[original_name].append(ann)

        for original_name in parts_ids.keys():
            images = api.image.download_nps(src_dataset.id, parts_ids[original_name])
            height = max_height[original_name]
            width = max_width[original_name]
            channels = images[0].shape[2]
            final_image = np.zeros((height, width, channels), dtype=np.uint8)
            final_ann = sly.Annotation(final_image.shape[:2])
            for image_part, ann, (top, left) in zip(images, parts_anns[original_name], parts_top_left[original_name]):
                def _translate_label(label):
                    shift_y = top
                    shift_x = left
                    return [label.translate(shift_y, shift_x)]

                ann = ann.transform_labels(_translate_label, new_size=final_image.shape[:2])
                final_image[top:top + image_part.shape[0], left:left + image_part.shape[1], :] = image_part
                final_ann = final_ann.add_labels(ann.labels)
                final_ann = final_ann.clone(img_tags=final_ann.img_tags.merge_without_duplicates(ann.img_tags))
            merged_image_info = api.image.upload_np(dst_dataset.id, original_name, final_image)
            api.annotation.upload_ann(merged_image_info.id, final_ann)
            progress.iters_done_report(len(images))

    api.task.set_output_project(task_id, dst_project.id, dst_project.name)
    app.stop()


def main():
    app.run(initial_events=[{"command": "merge"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
