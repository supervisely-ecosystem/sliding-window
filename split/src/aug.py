import cv2
import random
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from ast import literal_eval
import supervisely_lib as sly
import albumentations as A

aug_color_fg = None
aug_spacial_fg = None


# imgaug
# name_func_color = {
#     "GaussianNoise": iaa.imgcorruptlike.GaussianNoise,
#     "GaussianBlur": iaa.imgcorruptlike.GaussianBlur,
#     "GammaContrast": iaa.GammaContrast,
#     "Contrast": iaa.imgcorruptlike.Contrast,
#     "Brightness": iaa.imgcorruptlike.Brightness
# }

name_func_color = {
    "RandomBrightnessContrast": A.RandomBrightnessContrast,
    "CLAHE": A.CLAHE,
    "Blur": A.Blur,
}


name_func_spacial = {
    "Fliplr": iaa.Fliplr,
    "Flipud": iaa.Flipud,
    "Rotate": iaa.Rotate,
    #"ElasticTransformation": iaa.ElasticTransformation,
    "Resize": iaa.Resize,
}


def init_fg_augs(settings):
    init_color_augs(settings['objects']['augs']['color'])
    init_spacial_augs(settings['objects']['augs']['spacial'])


def init_color_augs(data):
    global aug_color_fg
    augs = []
    for key, value in data.items():
        for key, value in data.items():
            if key not in name_func_color:
                sly.logger.warn(f"Aug {key} not found, skipped")
                continue
        augs.append(name_func_color[key]())
    aug_color_fg = A.Compose(augs)


def init_spacial_augs(data):
    global aug_spacial_fg
    augs = []
    for key, value in data.items():
        if key == 'ElasticTransformation':
            alpha = literal_eval(value['alpha'])
            sigma = literal_eval(value['sigma'])
            augs.append(iaa.ElasticTransformation(alpha=alpha, sigma=sigma))
            continue
        if key not in name_func_spacial:
            sly.logger.warn(f"Aug {key} not found, skipped")
            continue

        parsed_value = value
        if type(value) is str:
            parsed_value = literal_eval(value)

        if key == 'Rotate':
            a = iaa.Rotate(rotate=parsed_value, fit_output=True)
        else:
            a = name_func_spacial[key](parsed_value)
        augs.append(a)
    aug_spacial_fg = iaa.Sequential(augs, random_order=True)


def apply_to_foreground(image, mask):
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image ({image.shape}) and mask ({mask.shape}) have different resolutions")

    # apply color augs
    augmented = aug_color_fg(image=image, mask=mask)
    image_aug = augmented['image']
    mask_aug = augmented['mask']

    # apply spacial augs
    segmap = SegmentationMapsOnImage(mask_aug, shape=mask_aug.shape)
    image_aug, segmap_aug = aug_spacial_fg(image=image_aug, segmentation_maps=segmap)
    mask_aug = segmap_aug.get_arr()
    return image_aug, mask_aug


def find_origin(image_shape, mask_shape):
    mh, mw = mask_shape[:2]
    ih, iw = image_shape[:2]
    if mh > ih or mw > iw:
        raise NotImplementedError("Mask is bigger that background image")

    x = random.randint(0, iw - mw)
    y = random.randint(0, ih - mh)
    return (x, y)


def resize_foreground_to_fit_into_image(dest_image, image, mask):
    img_h, img_w, _ = dest_image.shape
    mask_h, mask_w, _ = mask.shape

    settings = None
    if mask_h > img_h:
        settings = {
            "height": img_h,
            "width": "keep-aspect-ratio"
        }
    if mask_w > img_w and mask_w / img_w > mask_h / img_h:
        settings = {
            "height": "keep-aspect-ratio",
            "width": img_w
        }

    if settings is not None:
        aug = iaa.Resize(settings)
        segmap = SegmentationMapsOnImage(mask, shape=mask.shape)
        image_aug, segmap_aug = aug(image=image, segmentation_maps=segmap)
        mask_aug = segmap_aug.get_arr()
        return image_aug, mask_aug
    else:
        return image, mask


def place_fg_to_bg(fg, fg_mask, bg, x, y):
    sec_h, sec_w, _ = fg.shape
    secondary_object = cv2.bitwise_and(fg, fg_mask)
    secondary_bg = 255 - fg_mask
    bg[y:y+sec_h, x:x+sec_w, :] = cv2.bitwise_and(bg[y:y+sec_h, x:x+sec_w, :], secondary_bg) + secondary_object