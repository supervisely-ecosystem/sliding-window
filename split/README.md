<div align="center" markdown>
<img src="![sliding window split](https://user-images.githubusercontent.com/48245050/182573050-398ed69f-059f-49ba-b549-c3e1437fc839.png)
"/>

# Sliding Window Split

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
    <a href="#Screenshots">Screenshots</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/sliding-window/split)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/sliding-window)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/sliding-window/split)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/sliding-window/split)](https://supervise.ly)

</div>

# Overview

App splits all images and their labels using sliding window approach. Play with sliding window configuration and preview results before start splitting. All results will be saved to a new project. All crops can be mered back with another [Sliding window merge](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fsliding-window%252Fmerge) app. 

Industries and use cases: 
- microscopic images 
- satellite images
- images in huge resolution: defects detection, quality assurance in production
- split images for labelers 
- split images for neural network inference

# How To Use

<a data-key="sly-embeded-video-link" href="https://youtu.be/wbxXPyW5pLA" data-video-code="wbxXPyW5pLA">
    <img src="https://i.imgur.com/MS4dkKi.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
</a>

All images and their annotations will be split and saved to a new project. Also sliding window configuration is saved in project info. Just go to project -> `Info` tab.

<img src="https://i.imgur.com/usAjOiM.png"/>

Cropped image names are in the following format: 

`<image name>___<window index>_<window top coordinate>_<window left coordinate>.<image extension>`

for example:

`IMG_0748___0004_288_480.jpeg`
- image name: `IMG_0748`
- window index with leading zeros: `0004`
- window top coordinate: `288`
- window left coordinate: `480`
- image extension: `jpeg`

Such naming allows to perform opposite operation: merge all crops and labels to a single image another app.


# Screenshots

<img src="https://i.imgur.com/RXTCyNs.png"/>

