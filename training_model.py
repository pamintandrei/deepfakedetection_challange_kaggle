import cv2 as cv
from os import listdir
from os.path import isfile, join
from PIL import Image as pil_image
from transform import xception_default_data_transforms
import dlib
import time
import torch
import csv
import torch.nn as nn
import torch.optim as optim

model = torch.load('D:\\deepfake\\network\\all_c23.p')
model=model.cuda()
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
running_loss = 0.0



home='D:\deepfake\\test'
videos=listdir(home)

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb
def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output

def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

face_detector = dlib.get_frontal_face_detector()
model = torch.load('D:\\deepfake\\network\\all_c23.p')
model=model.cuda()
start_time=time.time()
counter=0
for each in videos:
    counter+=1
    if(counter%10==0):
        print(counter)
    if(counter==100):
        break
    nume=each
    each=home+'\\'+each

    capture = cv.VideoCapture(each)
    maxi=-1.0
    suma=0
    for i in range(0, 300):
        ret = capture.grab()

        if i % 30 == 0:
            ret, frame = capture.retrieve()

            if(ret):
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                faces = face_detector(gray, 1)
                if len(faces):
                    
                    faces=faces[0]
                    height, width = frame.shape[:2]
                    x, y, size = get_boundingbox(faces, width, height)
                    cropped_face = frame[y:y+size, x:x+size]
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward + backward + optimize
                    prediction,outputs = predict_with_model(cropped_face, model)
                    model.loss = criterion(outputs, ['1'])
                    model.loss.backward()
                    optimizer.step()
                    
                    # print statistics
                    running_loss += model.loss.item()
                    print(running_loss)

            # do something with frame


    capture.release()
    break

input()
# zero the parameter gradients
