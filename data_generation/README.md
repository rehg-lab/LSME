# CRIB-plusplus

Improved Infant Inspired Synthetic Data Generator

## Download Data

A blend file containing all Toys200 files, pbr materials and hdr backgrounds can be downloaded from

```
https://www.dropbox.com/s/5ndb8g34s0bey7j/objects.blend
```

Download the `hdr` and `pbr` files and place them in `common/assets` in directories named `pbr` and `hdr`

```
https://www.dropbox.com/s/9yoscwvzbrexqc3/pbr.tar
https://www.dropbox.com/s/oi5e29fra3y194i/hdr.tar
```

Download the following tarballs and untar them in `common`

```
https://www.dropbox.com/s/j6lsb8vkg051ks8/toys200_poses_canonical.tar
https://www.dropbox.com/s/nwts0kypxk7ntur/toys200_scene_configs.tar
```

Download the `hdr` and `pbr` files and place them in `common/assets` in directories named `pbr` and `hdr`

## Download Blender 2.93 LTS

```
https://www.blender.org/download/releases/2-93/
```

Once you adjust the paths in `rendering/render_toys200.sh` you should be able to get a quick demo to make sure things are working.

# How the code works
## 1. Object pose sampling
The first thing we do is "drop" all the objects on a plane to get their pose when they obey gravity. The code to do this is located in `object_pose_sampling`. For Toys200 use the bash script `sample_toys200.sh`. The provided `toys200_poses_canonical.tar` contains 4 different poses per object where the object was only slightly rotated before dropping (objects shouldn't end up totally upside down). This means that we don't have to do any rigid body simulations later and only rendering.
## 2. Scene config generation
The code for this is located in `scene_config_generation`. We want to generate a `.json` file for each scene we want to generate, picking the objects, their pose, the lighting environment and floor, and the camera pose all ahead of time before rendering. A sample script to do this is `toys200_create_scene_configs.py`.
## 3. Rendering
All we have to do here is just run `render_toys200.sh` with the appropriate paths. 
