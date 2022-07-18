import numpy as np
import cv2 , json
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile
from PIL import Image


app = FastAPI()


@app.get("/", tags=["Root"])
def get_root():
    return {"message": "Thank you for visiting this APP!😃"}

@app.post("/face_extractor", tags=["face_extractor"])
#: List = Query(None)
async def face_extractor(pic: UploadFile = File(...)):
    #extract_face using Blazeface(Mediapipe)
    mp_face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5
                    )
    mp_drawing = mp.solutions.drawing_utils
    FACIAL_KEYPOINTS = mp.solutions.face_detection.FaceKeyPoint

    file_byts = pic.file.read()   #Converting upload image to bytes
    #converting file bytes to Array of bytes
    pic = np.asarray(bytearray(file_byts), dtype="uint8")
    pic = cv2.imdecode(pic, cv2.IMREAD_COLOR)   #decoding bytesarray to Image Array
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)  #Converting the image from BGR to RGB

    #IMAGE = Image.fromarray(pic,'RGB')
    IMAGE = np.asarray(pic)
    image = IMAGE.copy()
    image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # get results
    results = mp_face_detection.process(image_input)

    if not results.detections:
        print('No faces detected.')
    else:
        # iterate over each detection and draw on image
        for detection in results.detections: 
            mp_drawing.draw_detection(image, detection)

    detection_results = []
    height=image.shape[0]
    width=image.shape[1]

    for detection in results.detections:
        # Landmarks
        keypoints = {}
        for kp in FACIAL_KEYPOINTS: # iterate over each landmarks and get from results
            keypoint = mp.solutions.face_detection.get_key_point(detection, kp)
            # convert to pixel coordinates and add to dictionary
            keypoints[kp.name] = {"x" : int(keypoint.x * width), "y" : int(keypoint.y * height)}
        
        bbox = detection.location_data.relative_bounding_box
        bbox_points = {
            "xmin" : int(bbox.xmin * width),
            "ymin" : int(bbox.ymin * height),
            "xmax" : int(bbox.width * width + bbox.xmin * width),
            "ymax" : int(bbox.height * height + bbox.ymin * height)
        }
        detection_results.append({
                "keypoints" : keypoints,
                "bbox" : bbox_points,
                "score" : detection.score
            })

    xleft = detection_results[0]["bbox"]["xmin"]
    xtop = detection_results[0]["bbox"]["ymin"]
    xright = detection_results[0]["bbox"]["xmax"]
    xbottom = detection_results[0]["bbox"]["ymax"]

    detected_faces = [(xleft, xtop, xright, xbottom)]
    image = IMAGE.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #cropping images
    face_array=np.array([])
    for n, face_rect in enumerate(detected_faces):
        face = Image.fromarray(image).crop(face_rect)
        face_np = np.asarray(face)
        image = Image.fromarray(face_np)
        image=image.resize((160,160))
        face_array = np.asarray(image)
        face = face_array.tolist()

    return json.dumps({
            "data": face,
            "status_code": 200,
            "message": "Successfully retrieved face data!"
        })
