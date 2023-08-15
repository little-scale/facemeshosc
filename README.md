# facemeshosc
MediaPipe Facemesh OSC

Based on https://github.com/cansik/mediapipe-osc as well as https://developers.google.com/mediapipe/solutions/vision/face_landmarker#get_started 

facemesh_osc.py sends all 468 facemesh landmarks via OSC
facemesh_osc_reduced.py sends a reduced set of landmarks via OSC (set the list in script)

The OSC message takes the form of /mediapipe/facemesh [landmark x][landmark y][landmark x][landmark y]...
