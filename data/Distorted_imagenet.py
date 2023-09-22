from dis import dis
import os
import json
import torch.utils.data as data
import numpy as np
from PIL import Image
from utils import distort_image, distort, undistort
import random
import warnings
import torch 
import torchvision.transforms as T
from glob import glob
import pickle as pkl
import torch.nn as nn
from datetime import datetime
import random


warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

pil = T.ToPILImage()
t = []

t.append(T.ToTensor())
# t.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
trans =  T.Compose(t)


def random_1():
    return 1 if random.random() < 0.5 else -1



def random_direction_normal(dim, n):
    p = np.abs(np.random.standard_normal((dim, n)))
    norm = np.sqrt(np.sum(p**2, axis=0))
    return p / norm

def random_direction_uniform(dim, n):
    p = np.random.uniform(0, 1, (dim, n))
    norm = np.sqrt(np.sum(p**2, axis=0))
    return p / norm

def random_magnitude_uniform(points, high=1):
    scale = np.random.uniform(0, high, points.shape[1])
    return points * scale

def random_magnitude_custom(points, high=1):
    scale = np.random.uniform(0, high, points.shape[1])**(1/points.shape[0])
    return points * scale




class M_distort(data.Dataset):
    def __init__(self, root, distortion, low, high, transform=None, xi=0.0, xi_bool = True, DA = True, task='train', target_transform=None, img_size=(64, 64)):
        super(M_distort, self).__init__()
        self.data_path = root
        self.xi_bool = xi_bool
        self.xi = xi
        self.img_size = img_size
        self.dist = distortion
        self.low = low
        self.high = high 
        self.DA = DA
        with open(self.data_path + '/classes.pkl', 'rb') as f:
            classes = pkl.load(f)
        
        self.classes = classes
        if task == 'train':
            with open(self.data_path + '/train/train.pkl', 'rb') as f:
                data = pkl.load(f)
            # with open(self.data_path + '/train_data.pkl', 'rb') as f:
            #     data = pkl.load(f)
        elif task == 'val':
            with open(self.data_path + '/val/val.pkl', 'rb') as f:
                data = pkl.load(f)
        elif task == 'test_1' or task == 'test_2' or task == 'test_3' or task == 'test_4' or task == 'test_5':
            with open(self.data_path + '/test_cls/test_cls.pkl', 'rb') as f:
                data = pkl.load(f)
            
            if distortion == 'spherical':
                with open(self.data_path + '/test_cls/' + task  + '.pkl', 'rb') as f:
                    test_dist = pkl.load(f)
                    self.test_dist = test_dist

        self.task = task 
        # self.data_img = data_img
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.

            return sampling points for each image and targets :) 
        """
        images = Image.open(self.data_path + '/' + self.data[index])

        if self.dist == 'polynomial' or self.dist == 'polynomial_woodsc':
            if self.task == 'train' or self.task == 'val':
                D = random.choice(self.wood_test)
            elif self.task == 'test_1' or self.task == 'test_2' or self.task == 'test_3' or self.task == 'test_4' or self.task == 'test_5':
                D = random.choice(self.wood_test)
            images = distort_image(images, D)


        elif self.dist == 'spherical':
            if self.task == 'train' or self.task == 'val':
                xi = random.uniform(self.low, self.high) 

            elif self.task == 'test_1' or self.task == 'test_2' or self.task == 'test_3' or self.task == 'test_4' or self.task == 'test_5' or self.task == 'test':
                if self.xi_bool:
                    xi = self.xi
                else:
                    xi = self.test_dist[self.data[index]]

            images, new_f, new_xi, new_fov  = distort(images, xi, f=9, im_size=self.img_size)
            images = Image.fromarray(images)
            if not self.DA:
                new_xi = 0.0
                new_f = 9/6

            D = np.array([new_xi, new_f, new_fov]).astype(np.float32)


        target = int(self.classes.index(self.data[index].split('/')[1]))


        if self.transform is not None:
            try:
                images = self.transform(images)
            except:
                print(images.size)
        # images = mask1*images
        return images, target, D, self.data[index]

    def __len__(self):
        return len(self.data)

if __name__=='__main__':
    # profiler.start()
    m = M_distort('/home-local2/akath.extra.nobkp/imagenet_2010', task='val', distortion='polynomial_woodsc', low = 0.5, high = 0.7, transform=trans)
    import pdb;pdb.set_trace()
    img = m[1]
    img = pil(img[0])
    img.save('undistort.png')
    # profiler.stop()
    # profiler.print()
    

