import numpy as np
import scipy as sp
from PIL import Image
import argparse

def main(args):
    array : np.ndarray = np.asarray(Image.open(args.path))
    print(Image.open(args.path).entropy())
    transformed = sp.fftpack.dct(array-128*np.ones(shape=array.shape,dtype=float))
    Image.fromarray(transformed,mode="RGB").save('transformed_out.png')
    Image.fromarray(sp.fftpack.idct(transformed)+128*np.ones(shape=array.shape,dtype=float),mode="RGB").save('transformed_inversed_out.png')
    quantized = sp.cluster.vq.whiten(transformed)
    Image.fromarray(quantized,mode="RGB").save('quantized_out.png')
    print(Image.fromarray(quantized,mode="RGB").entropy())

    img : Image = Image.fromarray(array,mode="RGB")
    img.save('out.png')

def load_rgb(path):
    with Image.open(path) as img:
        print(img.getchannel('R').getpixel((50,50)))

def load_channel_pixel_by_pixel(path,channel = 'R'):
    with Image.open(path) as img:
        print(img.size)
        array = np.ndarray(shape=img.size,dtype=int)
        for i in np.arange(0,img.size[0]):
            for j in np.arange(0,img.size[1]):
                value = 0
                for l in channel:
                    value += img.getchannel(l).getpixel((i,j))
                array[i,j] = value
        return array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path',type=str)
    args = parser.parse_args()
    main(args)