# Virtual Face
This project aims to make use of commonly avaliable webcam based tracking to provide inputs to control a virtual avatar.

# Detection and Traking
[MediaPipe](https://google.github.io/mediapipe/) offers a holistic suit of tracking solutions, ranging from face mesh 
and hand gesture tracking, to full body pose and a combination of everything by combining the various models.

Virtual Face uses the face mesh to estimate the pose and position of the face, as well as landmarks around the eyes, 
mouth and irises to provide full control over the head and expression.

# Feature Abstraction
The pose and position of the head is calculated based on the relative and absolute position of the silhouette landmarks 
respectively.

The extent to which each eye is opened is calculated and described by the eye aspect ratio (EAR) using a variation the method described in https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf.

Similarly, the extent to which the mouth is opened is calcuated and described using the mouth aspect ratio (
MAR) described in https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8871097.

Eye tracking is estimated using the same landmarks used to calculate the EAR to define the boundary of the eye, and the position of the irises are described relative to the horizontal and vertical boundaries (relative to the face). Here it is assumed that both eyes are focused on the same object and a singular average value is used from both eyes.

# Outputs
The abstractions extracted from the detected landmarks are made available to other programs by starting a socket client which can feed the data to a listening server.

A [companion project](https://github.com/Yi-Jiahe/cv-controller) makes use of the outputs from this program to control a 3D object in Unity. 
