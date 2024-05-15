import cv2 as cv
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fashioniq import Vton


SOURCE_IMG = "C:\\Users\\ragia\\Downloads\\Fashion IQ\\Fashion-IQ\\src\\cmate\\tests\\kurta1.jpeg"
DEST_IMG = "C:\\Users\\ragia\\Downloads\\Fashion IQ\\Fashion-IQ\\src\\cmate\\tests\\m1.jpeg"

def visualize_vton():
    start_time = time.time()
    cloth_blender = Vton(SOURCE_IMG, DEST_IMG)
    final_img, _ = cloth_blender.apply_cloth()
    print("Time elapsed (seconds):", time.time()-start_time)
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    source_frame = plt.imread(SOURCE_IMG)
    plt.imshow(source_frame)
    plt.title("Source Image")

    plt.subplot(1,3,2)
    dest_frame = plt.imread(DEST_IMG)
    plt.imshow(dest_frame)
    plt.title("Dest Image")

    plt.subplot(1,3,3)
    plt.imshow(cv.cvtColor(final_img, cv.COLOR_BGR2RGB))
    plt.title("Final Image")
    
    plt.axis("off")
    plt.show()

visualize_vton()