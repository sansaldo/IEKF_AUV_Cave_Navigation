# Invariant Extended Kalman Filter for Localization in Underwater Caves

Introduction: This is a final research project for NA 568/EECS 568/ROB 530 MOBILE ROBOTICS: METHODS & ALGORITHMS WINTER 2021 at the University of Michigan. The goal of this project is to use Invariant Kalman filtering on data gathered from underwater cave systems for autonomous localization. This program uses the accelerator and gyroscope data for prediction, DVL, depth sensor and magnetometer for correction, and the 6 traffic cones for ground truth. 

You can see our final presentation video of our program [here](https://www.youtube.com/watch?v=r809SdeicR8&t=4s)

## Dataset

The program relied on:

* [UNDERWATER CAVES SONAR AND VISION DATA SET](https://cirs.udg.edu/caves-dataset/) for data gathering
![Alt text](https://github.com/xinyuma0214/imgae/blob/main/image-folder/CavesGoogleEarth2.jpg)

## about file

* [riekf](https://github.com/sansaldo/IEKF_AUV_Cave_Navigation/blob/stacked/riekf.py) - Initialization, prediction, correction for riekf algorithm
* [import_data](https://github.com/sansaldo/IEKF_AUV_Cave_Navigation/blob/stacked/import_data.py) - Import data from the caves dataset(CSV file)
* [problem_skeleton](https://github.com/sansaldo/IEKF_AUV_Cave_Navigation/blob/stacked/problem_skeleton.py) - Main function to run our code
* [plot_ekf_results](https://github.com/sansaldo/IEKF_AUV_Cave_Navigation/blob/stacked/plot_ekf_results.py) - Plot cones position
* [localization_metrics](https://github.com/sansaldo/IEKF_AUV_Cave_Navigation/blob/stacked/localization_metrics.py) - Contain two metrics to compare our work with visual odometry and SLAM


## Getting started

These instructions will get you a copy of the project up and running on your local machine for development purposes.

### Prerequisites

Python3, numpy, scipy, os, csv and some knowledge about command line tool (e.g. terminal)

### Running the code

In order to configure this project, please follow these steps:

1. Clone the repository onto your local system.
```
$ git clone https://github.com/sansaldo/IEKF_AUV_Cave_Navigation.git
```

2. The "problem_skeleton.py" file will initiate the program.
```
$ python3 problem_skeleton.py
```
:bulb: __Note: in your terminal, please change the directory to the IEKF_AUV_Cave_Navigation directory before you run the program__

## Result presentation

Below are the 3D and 2D plots of our approach (in blue) versus visual odometry and SLAM algorithm approach after applied the DVL, Depth sensor and magnetometer for correction step. 
The triangles indicate those predicted positions of each cone (the colors indicate the cone number from 1 through 6)

You can see that our approach largely matches the other two in shape.
![Alt text](https://github.com/xinyuma0214/imgae/blob/main/image-folder/pres_2d.png)
![Alt text](https://github.com/xinyuma0214/imgae/blob/main/image-folder/pres_3d.png)

:smiley: If applicable, we will apply ROS package in our future work, which enables online usage of data. Data processing is available on [import_data_ros.py](https://github.com/sansaldo/IEKF_AUV_Cave_Navigation/blob/stacked/import_data_ros.py)


## Authors

* **Samuel Ansaldo** - [sansaldo](https://github.com/sansaldo)
* **AJ Bull** - [BullAJ](https://github.com/BullAJ)
* **Alyssa Scheske** - [ascheske](https://github.com/ascheske)
* **Shane Storks** - [shanestorks](https://github.com/shanestorks)
* **Xinyu Ma** - [xinyuma](https://github.com/uukool)

## Acknowledgments

* Professor [Maani Ghaffari](https://www.maanighaffari.com/) and the instruction team of EECS 568 in winter 2021, whose code examples helped form the basis of this project.



