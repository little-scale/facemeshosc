import argparse
import cv2
import mediapipe as mp

from mediapipe.framework.formats import landmark_pb2
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

from utils import add_default_args, get_video_input


OSC_ADDRESS = "/mediapipe/facemesh"

# The reduced set of landmarks are: 
# nose, upper lip, lower lip, left corner of mouth, right corner of mouth, left eyebrow, right eyebrow, top of head, bottom of chin
reduced_list = [4, 13, 16, 57, 287, 65, 295, 10, 152];
res_list = []; 

def send_facemesh(client: udp_client,
			   detections: [landmark_pb2.NormalizedLandmarkList]):
	if detections is None:
	   # client.send_message(OSC_ADDRESS, 0)
		return
	
	# create message and send
	list = [];
	builder = OscMessageBuilder(address=OSC_ADDRESS)
	
	for detection in detections:
		for landmark in detection.landmark:
			list.append(landmark.x)
			list.append(landmark.y)
			
	for i in range(len(reduced_list)):
		res_list[i * 2] = list[reduced_list[i] * 2]
		res_list[(i * 2) + 1] = list[(reduced_list[i] * 2) + 1]
	
	for i in range(len(res_list)):
		builder.add_arg(res_list[i])
	
	msg = builder.build()
	client.send(msg)


def main(): 
	# read arguments
	parser = argparse.ArgumentParser()
	add_default_args(parser)
	args = parser.parse_args()

	# create osc client
	client = udp_client.SimpleUDPClient(args.ip, args.port)
	
	for i in reduced_list:
		res_list.append(0)
		res_list.append(0)
		
	print(res_list)
	
	mp_drawing = mp.solutions.drawing_utils
	mp_face_mesh = mp.solutions.face_mesh

	face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1)
	drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
	cap = cv2.VideoCapture(get_video_input(args.input))

	while cap.isOpened():
		success, image = cap.read()
		if not success:
			break

		# Flip the image horizontally for a later selfie-view display, and convert
		# the BGR image to RGB.
		image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
		# To improve performance, optionally mark the image as not writeable to
		# pass by reference.
		image.flags.writeable = False
		results = face_mesh.process(image)
		
		send_facemesh(client, results.multi_face_landmarks)

		# Draw the face mesh annotations on the image.
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		if results.multi_face_landmarks:
			for face_landmarks in results.multi_face_landmarks:
				mp_drawing.draw_landmarks(
			  	  image=image,
			  	  landmark_list=face_landmarks,
			   	 connections=mp_face_mesh.FACEMESH_CONTOURS,
			   	 landmark_drawing_spec=drawing_spec,
			   	 connection_drawing_spec=drawing_spec)
		cv2.imshow('MediaPipe FaceMesh', image)
		if cv2.waitKey(5) & 0xFF == 27:
			break
	face_mesh.close()
	cap.release()
	

if __name__ == "__main__":
    main()
