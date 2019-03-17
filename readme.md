Click [here](https://github.com/birlrobotics/birl_baxter/wiki/DOPE(Deep-Object-Pose-Estimate)) to the wiki(Actually it is similar).


EN|[CN](https://github.com/birlrobotics/birl_baxter/wiki/DOPE(Deep-Object-Pose-Estimation))

> We updated the code in our group, please check [here](https://github.com/birlrobotics/Deep_Object_Pose/tree/dev).(Remember to change the branch to `dev`)    

## Contents List
- [Summary](#Summary)  
- [Prerequisites](#Prerequisites)
- [Quickstart](#Quickstart)
- [Overview](#Overview)
- [Structure](#Structure)
- [Video](#Video)
- [FAQ](#FAQ)

## Summary
DOPE(Deep Object Pose Estimation)system is developed by NVIDIA Lab. You can use the official DOPE ROS package for detection and 6-DoF pose estimation of known objects from an RGB camera. The network has been trained on several YCB objects. For more details, see the [original GitHub repo](https://github.com/NVlabs/Deep_Object_Pose).			

However, based on the official DOPE ROS package, we fixed some codes according to the paper. Besides, we created launch file `dope_image_demo.launch`、`dope_camera_demo.launch` to quickly start the demo, `image_read.py` script to get the images from the folder and publish them, `feature_map_visualization.py` script to save the output of the neural network as a image. Moreover, lots of useful links were added through the main codes to help you understand the codes more easily.     
     
![DOPE DETECTION](https://raw.githubusercontent.com/NVlabs/Deep_Object_Pose/master/dope_objects.png)    
Any questions are welcome!

## Prerequisites
1. Ubuntu 16.04 with full ROS Kinetic install(tested,it may works on other systems);      
2. NVIDIA GPU Supported(tested);    
3. CUDA 8.0(tested);  
4. CUDNN 6.0(tested);    

  > To install CUDA && CUDNN, you can check [here](https://blog.csdn.net/Hansry/article/details/81008210).(This blog was writen in chinese. If you have ang questions, feel free to contact us.)*
## Quickstart  
 > Note: The name of workspace(*catkin_ws*) below should be changed into yours.    

### 1、Download the DOPE code    
   ```
    $ cd ~/catkin_ws/src      
    $ git clone https://github.com/birlrobotics/Deep_Object_Pose.git
    $ git checkout dev   
   ```     
### 2、Install dependencies
   ```
	$ cd ~/catkin_ws/src/dope     
    $ pip install -r requirements.txt
   ```
### 3、Build
   ```
	$ cd ~/catkin_ws   
	$ catkin_make
   ```
### 4、Running 
1. Edit config info (if desired) in `~/catkin_ws/src/dope/config/config_pose.yaml`
    * `topic_camera`: RGB topic to listen to
    * `topic_publishing`: topic name for publishing
    * `weights`: dictionary of object names and their weights path name, **comment out any line to disable detection/estimation of that object**
    * `dimension`: dictionary of dimensions for the objects  (key values must match the `weights` names)
    * `draw_colors`: dictionary of object colors  (key values must match the `weights` names)
    * `camera_settings`: dictionary for the camera intrinsics; edit these values to match your camera
    * `thresh_points`: Thresholding the confidence for object detection; increase this value if you see too many false positives, reduce it if  objects are not detected.   

2. Download the weights
	> **Download [the weights](https://drive.google.com/open?id=1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg)** and save them to the `weights` folder,  

2. Run launch file to start
	> To test in real time with camera:     

	```
	$ roslaunch dope dope_camera_demo.launch
	```
	> To test off-line with images:

	```
	$ roslaunch dope dope_image_demo.launch
	```
3. Visualize in RViz
	> `Add > Image`to view the raw RGB image or the image with cuboids overlaid by changing the topic;

4. Ros topics published 
	> `$ rostopic list`

     ```
     /dope/webcam_rgb_raw       # RGB images from camera   
     /dope/dimension_[obj_name] # dimensions of object     
     /dope/pose_[obj_name]      # timestamped pose of object   
     /dope/rgb_points           # RGB images with detected cuboids overlaid
     ```
     *Note:* `[obj_name]` is in {cracker, gelatin, meat, mustard, soup, sugar}

5. Train the model:
	> Please refer to `python train.py --help` for specific details about the training code.    

	```
	$ python train.py --data path/to/dataset --object soup --outf soup --gpuids 0 1 2 3 4 5 6 7 
	```
	> This will create a folder called `train_soup` where the weights will be saved after each 10 epochs. It will use the 8 gpus using pytorch data parallel. 

## Overview   

  > A brief overview of this work is following:      
  >> 1. Put captured image into a deep neural network to extract the feature.     
  >> 2. The output of the neural network is two target maps named `beilef map` and `affinity map`.     
  >> 3. In order to extract the individual objects from the belied maps and retrieve the pose of the object, they used greedy algorithm as well as the PnP algorithm to process the two target maps.        
   >> ![Literature Overview](https://i.imgur.com/rJMEkOn.gif)
	 
## Structure
> The structure of the repository are following:    
![](https://i.imgur.com/XPOUJGH.png)   
     
> The neural network used in this work are following:   
> ![Neural Network](https://i.imgur.com/vaPRxSG.gif)
## Video
[![](https://i.imgur.com/sHjC2E4.png)](http://v.youku.com/v_show/id_XNDAzMzc0MjA5Ng==.html?spm=a2hzp.8244740.0.0)  
> You can watch the official video [here](https://www.youtube.com/watch?v=yVGViBqWtBI&feature=youtu.be)
