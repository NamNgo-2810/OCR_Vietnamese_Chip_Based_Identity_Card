from sources import app, templates
from fastapi import Request, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sources.Models import models
from sources.Models.database import SessionLocal, engine, SQLALCHEMY_DATABASE_URL
import databases
import yolov5
import os
from PIL import Image
from sources.Controllers import utils
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import sources.Controllers.config as cfg
import cv2

""" ---- Setup ---- """
# Init Database
database = databases.Database(SQLALCHEMY_DATABASE_URL)
models.Base.metadata.create_all(bind=engine)


async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Startup database server before start app
@app.on_event("startup")
async def startup_database():
    await database.connect()


# Shutdown database sever after closed app
@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


# Init yolov5 model
CORNER_MODEL = yolov5.load(cfg.CORNER_MODEL_PATH)
CONTENT_MODEL = yolov5.load(cfg.CONTENT_MODEL_PATH)

# Set conf and iou threshold -> Remove overlap and low confident bounding boxes
CONTENT_MODEL.conf = cfg.CONF_CONTENT_THRESHOLD
CONTENT_MODEL.iou = cfg.IOU_CONTENT_THRESHOLD

# CORNER_MODEL.conf = cfg.CONF_CORNER_THRESHOLD
# CORNER_MODEL.iou = cfg.IOU_CORNER_THRESHOLD

# Config directory
UPLOAD_FOLDER = cfg.UPLOAD_FOLDER
SAVE_DIR = cfg.SAVE_DIR
FACE_CROP_DIR = cfg.FACE_DIR

""" Recognization detected parts in ID """
config = Cfg.load_config_from_name('vgg_seq2seq')  # OR vgg_transformer -> acc || vgg_seq2seq -> time
# config = Cfg.load_config_from_file(cfg.OCR_CFG)
# config['weights'] = cfg.OCR_MODEL_PATH
config['cnn']['pretrained'] = False
config['device'] = cfg.DEVICE
config['predictor']['beamsearch'] = False
detector = Predictor(config)


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/uploader")
async def upload(file: UploadFile = File(...)):
    INPUT_IMG = os.listdir(UPLOAD_FOLDER)
    if INPUT_IMG is not None:
        for uploaded_img in INPUT_IMG:
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))

    file_location = f"./{UPLOAD_FOLDER}/{file.filename}"
    contents = await file.read()
    with open(file_location, 'wb') as f:
        f.write(contents)

    # Validating file
    INPUT_FILE = os.listdir(UPLOAD_FOLDER)[0]
    if INPUT_FILE == 'NULL':
        os.remove(os.path.join(UPLOAD_FOLDER, INPUT_FILE))
        error = "No file selected!"
        return JSONResponse(status_code=403, content={"message": error})
    elif INPUT_FILE == 'WRONG_EXTS':
        os.remove(os.path.join(UPLOAD_FOLDER, INPUT_FILE))
        error = "This file is not supported!"
        return JSONResponse(status_code=404, content={"message": error})

    # return {"Filename": file.filename}
    return await extract_info()


@app.post("/extract")
async def extract_info(ekyc: bool = False, path_id: str = None):
    """Check if uploaded image exists and process it."""

    # Ensure the upload folder exists
    if not os.path.isdir(cfg.UPLOAD_FOLDER):
        os.mkdir(cfg.UPLOAD_FOLDER)

    INPUT_IMG = os.listdir(cfg.UPLOAD_FOLDER)
    if INPUT_IMG:
        img_path = os.path.join(cfg.UPLOAD_FOLDER, INPUT_IMG[0]) if not ekyc else path_id
    else:
        return JSONResponse(status_code=400, content={"message": "No images found in the upload folder."})

    # Step 1: Detect QR Code
    img = cv2.imread(img_path)

    # Detect corners
    CORNER = CORNER_MODEL(img)
    predictions = CORNER.pred[0]
    categories = predictions[:, 5].tolist()  # Class
    if len(categories) != 4:
        error = "Detecting corners failed!"
        return JSONResponse(status_code=401, content={"message": error})

    boxes = utils.class_Order(predictions[:, :4].tolist(), categories)  # x1, y1, x2, y2
    IMG = Image.open(img_path)
    center_points = list(map(utils.get_center_point, boxes))

    # Temporary fixing
    c2, c3 = center_points[2], center_points[3]
    c2_fix, c3_fix = (c2[0], c2[1] + 30), (c3[0], c3[1] + 30)
    center_points = [center_points[0], center_points[1], c2_fix, c3_fix]
    center_points = np.asarray(center_points)
    aligned = utils.four_point_transform(IMG, center_points)

    # Convert from OpenCV to PIL format
    aligned = Image.fromarray(aligned)

    # Detect content
    CONTENT = CONTENT_MODEL(aligned)
    predictions = CONTENT.pred[0]
    categories = predictions[:, 5].tolist()  # Class
    if 7 not in categories and len(categories) < 9:
        error = "Missing fields! Detecting content failed!"
        return JSONResponse(status_code=402, content={"message": error})
    elif 7 in categories and len(categories) < 10:
        error = "Missing fields! Detecting content failed!"
        return JSONResponse(status_code=402, content={"message": error})

    boxes = predictions[:, :4].tolist()

    # Non-Maximum Suppression
    boxes, categories = utils.non_max_suppression_fast(np.array(boxes), categories, 0.7)
    boxes = utils.class_Order(boxes, categories)  # x1, y1, x2, y2

    # Ensure save directory exists and is empty
    if not os.path.isdir(cfg.SAVE_DIR):
        os.mkdir(cfg.SAVE_DIR)
    else:
        for f in os.listdir(cfg.SAVE_DIR):
            os.remove(os.path.join(cfg.SAVE_DIR, f))

    # Save detected parts
    for index, box in enumerate(boxes):
        left, top, right, bottom = box
        if 5 < index < 9:
            right += 100
        cropped_image = aligned.crop((left, top, right, bottom))
        cropped_image.save(os.path.join(cfg.SAVE_DIR, f'{index}.jpg'))

    # Collect all detected parts
    FIELDS_DETECTED = []
    for idx, img_crop in enumerate(sorted(os.listdir(cfg.SAVE_DIR))):
        if idx > 0:
            img_ = Image.open(os.path.join(cfg.SAVE_DIR, img_crop))
            s = detector.predict(img_)
            FIELDS_DETECTED.append(s)

    if 7 in categories:
        FIELDS_DETECTED = FIELDS_DETECTED[:6] + [FIELDS_DETECTED[6] + ', ' + FIELDS_DETECTED[7]] + [FIELDS_DETECTED[8]]

    response = {
        "data": FIELDS_DETECTED
    }

    response = jsonable_encoder(response)
    return JSONResponse(content=response)
