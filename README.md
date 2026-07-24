# Overview
In this project I designed optimal controllers for a robotic bicycle, using deep reinforcement learning. I didn’t use any mathematical models during the controller design process. Moreover, the controller parameters were automatically tuned through the interactions which take place between the controller and the environment. The final controller was comprised of two separate controllers; One for balancing the bicycle and another for autonomous navigation. The bicycle was balanced by a PID controller. On the other hand, the autonomous navigation was made possible by both a PID controller and a nonlinear controller, and they were compared based on their performance. Finally, I demonstrated that the proposed nonlinear architecture can outperform the simple linear controller.

# Mechanism of the Robotic Bicycle
Real bicycles are composed of various parts, such as bearings, sprockets, wheels, handlebar, etc. However, in this project only those parts with more significant effect on the motion of bicycle are considered. These parts are as follows:
1. Frame
2. Handlebar
3. Front wheel
4. Rear wheel

Bicycle is an inherently unstable system and the dynamic equations governing its motion are highly nonlinear. In order to balance the bicycle, we can use a flywheel. At every time instant, the controller sends an actuating signal to the flywheel (a torque), and as a result, the flywheel exerts a reaction torque to the frame, and this process can keep the bicycle upright, provided that the controller is designed properly. Therefore, in this report, the robotic bicycle is considered as an assembly of five rigid links which are connected to each other by means of four revolute joints. Note that here the robotic bicycle is assumed as a system with three inputs:

1. Torque applied to the rear wheel
2. Torque applied to the handlebar
3. Torque applied to the flywheel

# Computer Modeling and Simulation
The 3D model of the robotic bicycle is designed using SOLIDWORKS. Then, each part of the bicycle is converted into the STL format, since MuJoCo can’t interpret the SOLIDWORKS part files directly. Other information regarding the configuration of the model, such as connections between the links, definitions of sensors and actuators, are specified in a separate XML file. Finally, these files are loaded by the main C++ program and the simulation is performed.

## 3D Model of the Robotic Bicycle in SOLIDWORKS
<img width="879" height="458" alt="image" src="https://github.com/user-attachments/assets/ca44ff80-66c2-4177-8a10-66b56dc49fe3" />



## Simulation of the Bicycle in MuJoCo
<img width="516" height="477" alt="17" src="https://github.com/user-attachments/assets/2ea425ee-e861-4041-8eef-c7fdf5ebb16b" />

## Coordinate Systems Attached to the Links of the Robotic Bicycle
<img width="676" height="505" alt="18" src="https://github.com/user-attachments/assets/1dcda925-fe0c-4ec5-ac59-b8eafa495c03" />

## Simulation of the Walls in the MuJoCo Environment
<img width="577" height="555" alt="10" src="https://github.com/user-attachments/assets/76433625-0db6-4643-98e1-b4c7c3bdcc2b" />
