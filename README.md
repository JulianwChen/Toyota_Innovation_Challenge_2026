## Toyota Innovation Challenge
Hosted by the University of Waterloo Engineering Ideas Clinic and Toyota Motor Manufacturing Canada(TMMC).
Toyota engineers introduced real problems they faced in manufacturing environments to students to solve.
Group: Julian Chen, Presley Prince, Chakib Belghait, Baran Mangoli and Ralph Kalunga


# Collaborative Robotic Arm 
## Challenge Overview
Robotics are growing increasingly common in manufacturing/warehouse settings to assist humans with various tasks. Traditionally, robots are in cages where human workers must stay out of during their operation. Toyota would like to take an innovative approach, where production workers and robots collaborate to achieve productivity boosts on their factory floor. This means that the robot and the human worker will be working on the same task but handling different parts of the task. 

You are tasked with designing the control of a desktop robotic arm that resembles a scaled-down version of a high torque arm found in TMMC, to work along with humans to achieve automative assembly tasks. Engineers at TMMC have already enabled the robot arm to complete a pick-and-place motion according to human actions. Your role is to continue developing this function to achieve more robot collaboration capabilities. The robot should consider efficiency, the safety of users around it, as well as the comfort of the users who interact with the robots.  

## Potential Solutions: 
- **Vision/Perception**: improve the current computer vision to achieve the detection of the car parts
- **Human-machine interface**: add in functions to detect worker's hand motions and steer robot away
- **Machine Controls**: improvement for robotic arm to accurately pick up car parts and place it in the corresponding box
- **Friendly Robots**: how should the robot behave to feel like a companion to workers instead of replacement
- **More areas to discover**: what do you think the current robotic arm should accomplish to collaborate human worke


## Starter Material Introduction:
Each team will be provided with the following equipment: 
1. A Dobot Magician arm with a gripper at the tip
2. A Orbbec camera
3. Camera Stand to mount camera
The image below shows an example setup of the camera and the robot arm. This setup allows the camera to see the robot arm's location as well as the table. You can modify this setup according to your design needs.

Setup: 

Clipping Camera Stand:

Each team will also be provided with 3 red tags to run the starter code. Teams can also request for 3 printed car parts as a part of their development.



## Must Read Before Use: Robot Arm Set Up & Safety Training
Please read through the safety operation procedure in [here](https://github.com/IdeasClinicUWaterloo/S26-toyota-innovation-challenge/tree/docs/Collaborative_Robotics/Safety%20Instructions) before operating the provided robot arm. Failure to do so may result in damage to personal property or even injuries.

## Set up: (rerun this if you run into issues where the error is unable to connect to robot)
1. ensure all peripherals are plugged in: robot arm, camera
2. power up the arm by pressing the power button, wait for its calibration to coplete
3. launch DobotLab, click "connect" to the arm. After connecting, verify the COM port the robot is connected to
  - if asked to select an arm model: select "Dobot magician"
  [image here]
4. in dobotArm.py, ctrl+f to find `state = dType.ConnectDobot(api, "COM7", 115200)[0]`. Change the number after COM to reflect on the port you find
5. disconnect from DobotLab
5. if you every unplugged the robot arm, rerun the steps above the confirm the communication port is correct



## Basic Robot Control
Refer to `testDobot.py` for basic Dobot control codes

**Notes**

- Do not modify the lib folder (unless you're sure of what you're doing) -> it includes the DLLs to interface with DobotLink
- dobotArm.py contains the python wrapper functions. Try to call from this library unless absolutely needed. Only modify the functions inside if you know what you are doing



### Using Intrinsic Calibration Data

You need to first import all the data

```python
data          = np.load("camera_params.npz")
camera_matrix = data["camera_matrix"]   # 3x3 intrinsic matrix  (K)
dist_coeffs   = data["dist_coeffs"]     # distortion vector
```

- Option A - undistort a single frame (simple):
  `undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)`

- Option B - fast per-frame using pre-computed maps (recommended for live video):

```python
h, w = frame.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), alpha=1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs,
                                              None, new_K, (w,h), cv2.CV_16SC2)
undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
```

- Option C - pose estimation with solvePnP:
  `cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)`


