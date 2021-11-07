# Virtual Face
This project aims to make use of camera based tracking to provide inputs to control a virtual avatar.

# Detection and Traking
[MediaPipe](https://google.github.io/mediapipe/) offers a holistic suit of tracking solutions, ranging from face mesh 
and hand gesture tracking, to full body pose and a combination of everything by combining the various models.

Virtual Face uses the face mesh to estimate the pose and position of the face, as well as landmarks around the eyes, 
mouth and irises to provide full control over the head and expression.

# Feature Abstraction
The pose and position of the head is calculated based on the relative and absolute position of the silhouette landmarks 
respectively.

# Outputs
The estimations and abstractions extracted from the landmarks are made available to other programs by opening a socket 
for clients to connect to and request the data from.

A [companion project](https://github.com/Yi-Jiahe/cv-controller) makes use of the outputs from this program to control a 3D object in Unity. 