import threading
import cv2
import numpy as np
import random
import time
from tqdm import tqdm

class MyThreading(threading.Thread):

    def __init__(self, func, arg):
        super(MyThreading,self).__init__()
        self.func = func
        self.arg = arg

    def run(self):
        self.func(self.arg)
H = 50
W = 50
image = np.zeros((H,W,3),dtype=np.uint8)

def my_func(args):
    # Do heavy computation ...
    y = args[2]//W
    x = args[2]%W
    args[0].acquire()
    args[1][y,x] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    args[0].release()

lock = threading.Lock()
n_threads = 5
cur_pix = 0
threads_list =[]
for i in range(n_threads):
    obj = MyThreading(my_func, (lock,image,cur_pix))
    cur_pix+=1
    threads_list.append(obj)
    obj.start()
    # obj.join()

pbar = tqdm(total = H*W,desc = 'Progress',unit='pix')
while cur_pix< H*W:
    for t in range(n_threads):
        if not threads_list[t].is_alive():
            threads_list[t] = MyThreading(my_func, (lock,image,cur_pix))
            cur_pix+=1
            threads_list[t].start()
            # threads_list[t].join()
            pbar.update()

cv2.imshow('image',image)
cv2.waitKey(0)