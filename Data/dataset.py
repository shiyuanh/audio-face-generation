import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
import os
from PIL import Image
import random
import math
import cv2
import time
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

def ToTensor(pic):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backard compability
        return img.float().div(255)
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


# You should build custom dataset as below.
class CelebA(data.Dataset):
    def __init__(self,dataPath='data/CelebA/images/',loadSize=64,fineSize=64,flip=1):
        super(CelebA, self).__init__()
        # list all images into a list
        self.image_list = [x for x in listdir(dataPath) if is_image_file(x)]
        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        path = os.path.join(self.dataPath,self.image_list[index])
        img = default_loader(path) 
        w,h = img.size

        if(h != self.loadSize):
            img = img.resize((self.loadSize, self.loadSize), Image.BILINEAR)

        if(self.loadSize != self.fineSize):
            #x1 = random.randint(0, self.loadSize - self.fineSize)
            #y1 = random.randint(0, self.loadSize - self.fineSize)
             
            x1 = math.floor((self.loadSize - self.fineSize)/2)
            y1 = math.floor((self.loadSize - self.fineSize)/2)
            img = img.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))

        if(self.flip == 1):
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = ToTensor(img) # 3 x 256 x 256

        img = img.mul_(2).add_(-1)
        # 3. Return a data pair (e.g. image and label).
        return img

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)

class VoxCeleb(data.Dataset):
    def __init__(self,dataPath='/home/linxd/gan/data/voxceleb/unzippedFaces',meta = 'voxceleb/speakers.txt', audio = 'voxceleb/feature_logits.txt', loadSize=64,fineSize=64,flip=1, crop = False):
        super(VoxCeleb, self).__init__()
        self.image_paths = list()
        self.audio = list()
        with open(audio, 'r') as f1:
            with open('voxceleb/audios.txt', 'r') as f2:
                fea = f1.readlines()
                names = f2.readlines()
                
        
        # list all images into a list
        a=0
        idx = 0
        with open(meta,'r') as f:
            for line in f:
                fields = line.strip().split(',')
                path = os.path.join(dataPath, fields[0], '1.6', fields[1])
                imgs = listdir(path)
                temp = []
                for x in imgs:
                    if is_image_file(x):
                        temp.append(int(x[:7]))
                if len(temp) > 0:
                    while idx < len(names):
                        
                        if not (names[idx].startswith(fields[0] + '/val/' + fields[1]) or names[idx].startswith(fields[0] + '/test/' + fields[1]) or \
                        names[idx].startswith(fields[0] + '/train/' + fields[1])):
                            idx = idx + 1
                        else:
                            break
                    seg = (fea[idx - 1].split(' '))[1].split(',')
                    for i in range(len(seg)):
                        seg[i] = float(seg[i])
                    self.audio.append(np.asarray(seg)) 
                    temp.sort()
                    first = temp[0]
                    self.image_paths.append(os.path.join(path, '{:07d}.jpg'.format(first)))
                    a+=1
                    self.image_paths.append(os.path.join(path, '{:07d}.jpg'.format(temp[1])))
                    a+=1
                    self.image_paths.append(os.path.join(path, '{:07d}.jpg'.format(temp[2])))
                    a+=1
                    self.image_paths.append(os.path.join(path, '{:07d}.jpg'.format(temp[3])))
                    a+=1
        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip
        self.crop = crop
        print('Audio list length', len(self.audio))
        print('Done Loading Dataset...', len(self.image_paths))
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        path = self.image_paths[index * 4 + np.random.choice(4,1)[0]]
        audio = self.audio[index]
        img = np.asarray(default_loader(path))
        
        if self.crop:
            cascPath = '/home/linxd/gan/began/Data/haarcascade_frontalface_default.xml'
            faceCascade = cv2.CascadeClassifier(cascPath)
            #print(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #print(gray)
            faces = faceCascade.detectMultiScale(gray, 5, 5)
            if len(faces) == 0:
                pass
            else:
                x,y,w_,h_ = faces[0]
                #img = img[y:y+max(w_, h_), x:x+max(w_, h_), :]
                img = img[y:y+w_, x:x+w_, :]
            img = Image.fromarray(img)
        w,h = img.size
        if(h != self.loadSize):
            img = img.resize((self.loadSize, self.loadSize), Image.BILINEAR)

        if(self.loadSize != self.fineSize):
            #x1 = random.randint(0, self.loadSize - self.fineSize)
            #y1 = random.randint(0, self.loadSize - self.fineSize)
             
            x1 = math.floor((self.loadSize - self.fineSize)/2)
            y1 = math.floor((self.loadSize - self.fineSize)/2)
            img = img.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))

        if(self.flip == 1):
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = ToTensor(img) # 3 x 256 x 256

        img = img.mul_(2).add_(-1)
        # 3. Return a data pair (e.g. image and label).
        return img, audio.astype(np.float32)

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.audio)
    
class VoxCeleb2(data.Dataset):
    def __init__(self,dataPath='/home/linxd/gan/data/voxceleb/unzippedFaces', audiofcPath = '/home/linxd/gan/data/url_audio_features', meta='/home/shiyuan/BEGAN/voxceleb/track_info/speakers_split.txt',loadSize=128, fineSize=128, flip=1, crop = False):
        super(VoxCeleb2, self).__init__()
        self.image_paths = list()
        self.audio_features = list()
        
        with open(meta,'r') as f:
            name_id_urls = f.readlines()
        name_id_urls = [x.strip().split() for x in name_id_urls]
        name_id_urls = [x[:-1] for x in name_id_urls if x[-1]=='train']
        print(len(name_id_urls))
        
        t1=time.time()
        for name_id_url in name_id_urls:
            name,idx,url = name_id_url
            name_id = '_'.join([name,idx])
            audiofc_file = os.path.join(audiofcPath, name_id, url,'fc.txt')
            frame = np.loadtxt(audiofc_file, delimiter = ' ', dtype = str, usecols = 0)
            audiofc = np.loadtxt(audiofc_file, delimiter = ' ', dtype = str, usecols = 1)
            
            frame = [x.strip().strip('b').strip("'").split(',') for x in frame]
            frame_path = [os.path.join(dataPath, x[0],'1.6',x[2],x[-1]+'.jpg') for x in frame]            
            audiofc = [list(map(float, x.strip().strip('b').strip("'").split(','))) for x in audiofc]

            
            '''
            for f, af in zip(frame_path, audiofc):
                if os.path.isfile(f):
                    self.image_paths.append(f)
                    self.audio_features.append(af)
                else:
                    pass
                    #print(f)
            
            '''
            self.image_paths.extend(frame_path)
            self.audio_features.extend(audiofc)
            


            
        t2=time.time()
        print(t2-t1)

        self.audio_features = np.asarray(self.audio_features)
        #self.audio_features = (self.audio_features - np.mean(self.audio_features, axis = 0))
        #self.audio_features = self.audio_features / np.max(np.abs(self.audio_features), axis = 0)
        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip
        self.crop = crop
                
        print('Audio list length', self.audio_features.shape)
        print('Done Loading Dataset...', len(self.image_paths))
        
        
        
        
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        
        succ = False
        #print(index)
        while not succ:
            #print(index,'in')
            try:
                path = self.image_paths[index]
                #print(path)
                audio = np.array(self.audio_features[index,:].tolist())
                #print(audio)
                img = np.asarray(default_loader(path))

                #print(img, path)
                if self.crop:
                    cascPath = '/home/linxd/gan/began/Data/haarcascade_frontalface_default.xml'
                    faceCascade = cv2.CascadeClassifier(cascPath)
                    #print(img)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    #print(gray)
                    faces = faceCascade.detectMultiScale(gray, 5, 5)
                    if len(faces) == 0:
                        pass
                    else:
                        x,y,w_,h_ = faces[0]
                        #img = img[y:y+max(w_, h_), x:x+max(w_, h_), :]
                        img = img[y:y+w_, x:x+w_, :]

                img = Image.fromarray(img)
                w,h = img.size
                if(h != self.loadSize):
                    img = img.resize((self.loadSize, self.loadSize), Image.BILINEAR)

                if(self.loadSize != self.fineSize):
                    #x1 = random.randint(0, self.loadSize - self.fineSize)
                    #y1 = random.randint(0, self.loadSize - self.fineSize)

                    x1 = math.floor((self.loadSize - self.fineSize)/2)
                    y1 = math.floor((self.loadSize - self.fineSize)/2)
                    img = img.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))

                if(self.flip == 1):
                    if random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        
                w,h=img.size

                
                img = ToTensor(img) # 3 x 256 x 256
                #print(audio.shape,audio.shape == (256,),w==self.loadSize,h==self.loadSize)
                if audio.shape == (256,) and w==self.loadSize and h==self.loadSize:
                    succ = True
                else:
                    index = random.randint(0, self.__len__()-1)
                    print('warning: auio dimension wrong ', index, path)
                    try:
                        f=open('bad_audio.txt','a')
                        f.write(path+'\n')
                        f.close()
                    except:
                        pass
            
            except Exception as e:
                index = random.randint(0, self.__len__()-1)
                try:
                    f=open('nonexist_frame.txt','a')
                    print('warning: nonexist_frame ',index, path)
                    f.write(path+'\n')
                    f.close()
                except:
                    pass
        # 3. Return a data pair (e.g. image and label).
        return img, audio.astype(np.float32)

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_paths)
    
class VoxCeleb3(data.Dataset):
    def __init__(self,dataPath='/home/linxd/gan/data/voxceleb/unzippedFaces',meta = '/home/shiyuan/BEGAN/voxceleb/track_info/speakers_split.txt', audio = '/home/shiyuan/VGGVox/audiofc_1024.txt', loadSize=64,fineSize=64,flip=1, crop = False):
        super(VoxCeleb3, self).__init__()
        self.image_paths = list()
        self.audio_features = list()
        with open(audio, 'r') as f1:
            audiofc = f1.readlines()
        
        with open(meta, 'r') as f2:
            speakers = f2.readlines()

        self.audio_features = np.asarray([list(map(float, x.strip().split(' '))) for x in audiofc])

        
        for sp in speakers:
            sp = sp.strip().split(' ')
            path = os.path.join(dataPath, sp[0], '1.6', sp[2])
            self.image_paths.append(path)
        
        # list all images into a list
        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip
        self.crop = crop
        print('Audio list length', self.audio_features.shape)
        print('Done Loading Dataset...', len(self.image_paths))
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        succ = False
        while not succ:
            try:
                path = self.image_paths[index]
                audio = np.array(self.audio_features[index, :].tolist())
                imgs = os.listdir(path)
                if len(imgs) > 0:
                    path = os.path.join(path, imgs[0])
                img = np.asarray(default_loader(path))


                if self.crop:
                    cascPath = '/home/linxd/gan/began/Data/haarcascade_frontalface_default.xml'
                    faceCascade = cv2.CascadeClassifier(cascPath)
                    #print(img)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    #print(gray)
                    faces = faceCascade.detectMultiScale(gray, 5, 5)
                    if len(faces) == 0:
                        pass
                    else:
                        x,y,w_,h_ = faces[0]
                        #img = img[y:y+max(w_, h_), x:x+max(w_, h_), :]
                        img = img[y:y+w_, x:x+w_, :]

                img = Image.fromarray(img)
                w,h = img.size
                if(h != self.loadSize):
                    img = img.resize((self.loadSize, self.loadSize), Image.BILINEAR)

                if(self.loadSize != self.fineSize):
                    #x1 = random.randint(0, self.loadSize - self.fineSize)
                    #y1 = random.randint(0, self.loadSize - self.fineSize)

                    x1 = math.floor((self.loadSize - self.fineSize)/2)
                    y1 = math.floor((self.loadSize - self.fineSize)/2)
                    img = img.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))

                if(self.flip == 1):
                    if random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        
                w,h=img.size

                
                img = ToTensor(img) # 3 x 256 x 256
                if audio.shape == (1024,) and w==self.loadSize and h==self.loadSize:
                    succ = True
                else:
                    index = random.randint(0, self.__len__()-1)
                    try:
                        f=open('bad_audio.txt','a')
                        f.write(path+'\n')
                        f.close()
                    except:
                        pass
            
            except Exception as e:
                index = random.randint(0, self.__len__()-1)
                try:
                    f=open('nonexist_frame.txt','a')
                    f.write(path+'\n')
                    f.close()
                except:
                    pass
        # 3. Return a data pair (e.g. image and label).
        return img, audio.astype(np.float32)

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_paths)