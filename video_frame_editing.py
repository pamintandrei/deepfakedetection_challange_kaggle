import cv2 as cv
from os import listdir
from os.path import isfile, join
from PIL import Image 
from PIL import ImageFilter as imf
from transform import xception_default_data_transforms
import os
import dlib
import time
import numpy as np
home='E:\\dfdc_train_part_3\\' #unde ai videouri
videos=os.listdir(home) #lista cu videouri
for each in videos:
    if(each!='metadata.json'):
        name=each[:-4] #elimina .mp4 din nume ca sa il salvez cu .jpg
    
        each=home+each #scrie numele complet al videoului cu directoryul
        try:
            capture = cv.VideoCapture(each) #deschide videoul
            ret = capture.grab() #ia primul frame 
            ret2, frame = capture.retrieve() #decodeaza primul frame
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)#transforma din BGR in RGB pentru ca PIL lucreza cu RGB
            pil_frame=Image.fromarray(frame) # fa-mi un pil image din frame
            lista=[] #prima lista de frameuri
            listablend1=[]#a doua lista de frameuri in care dau si blend
            blendall=Image.new("RGB",pil_frame.size,color=1) #creez o imagine de baza pt blend
            #for ca sa ia fiecare frame
            for i in range(0, 300):
                ret = capture.grab() 
                if(i%10==0):
                    if(ret):
                        ret1=True#nu are legatura cu nimic e legacy code
                        prevframe=frame #same
                        ret2, frame = capture.retrieve()
                        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                        pil_frame=Image.fromarray(frame,"RGB")#transform arrayul frameului intr-un PIL image
                        pil_frame=pil_frame.filter(imf.SMOOTH_MORE)#pun un filtru de smooth 
                        lista.append(pil_frame)#pun in lista frameul
                        '''
                        blendall=Image.blend(blendall,pil_frame,0.5)
                        '''
            '''
                if(ret1 and ret2):
                    
                    pil_prevframe=Image.fromarray(prevframe)
                    mask = Image.new("L", pil_prevframe.size, 128)
                    pil_frame=Image.fromarray(frame)
                    composite=Image.blend(pil_prevframe,pil_frame,0.5)
                    composite.show()
                    
                    # do something with frame
                ret2=False
            '''
            #dau blend in 2 cate 2 ,0.5 zice sa ia jumate din primu si jumate din al doilea
            for i in range(0,28,2):
                blendall=Image.blend(lista[i],lista[i+1],0.5)
                listablend1.append(blendall)
            #dau blend in la toata in blendall
            for i in range(0,14,1):
                blendall=Image.blend(listablend1[i],blendall,0.5)
            #dau smooth si sharpen ca sa mai elimin din artefacte
            blendall=blendall.filter(imf.SMOOTH_MORE)
            blendall=blendall.filter(imf.SMOOTH_MORE)
            blendall=blendall.filter(imf.SHARPEN)
            blendall=blendall.filter(imf.SHARPEN)
            blendall=blendall.filter(imf.SHARPEN)
            blendall=blendall.filter(imf.SHARPEN)
            blendall.save('E:\\blend_part_3\\'+name+'.jpg')
        except:
            pass