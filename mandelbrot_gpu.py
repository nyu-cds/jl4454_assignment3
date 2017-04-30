
# 
# A CUDA version to calculate the Mandelbrot set
#
from numba import cuda
import numpy as np
from pylab import imshow, show

@cuda.jit(device=True)
def mandel(x, y, max_iters):
    '''
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the 
    Mandelbrot set given a fixed number of iterations.
    '''
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@cuda.jit
def compute_mandel(min_x, max_x, min_y, max_y, image, iters):
    '''
     Follow the question requirements to modify the compute_mandel function assigns a value
     to each element of the image array:
     1.first, as the initial program, calculate pixel_size_x and pixel_size_y
     2.as required, obtain the starting x and y coordinates using cuda.grid()
     3.obtain the size of the block using gridDim and blockDim
     4.get the range of x and y
     5.calculate the mandel value for each element of the block
    '''
    #first, as the initial program, calculate pixel_size_x and pixel_size_y
    height = image.shape[0]
    width = image.shape[1]
    
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    
    #as required, obtain the starting x and y coordinates using cuda.grid()
    x0, y0=cuda.grid(2)
    
    #obtain the size of the block using gridDim and blockDim
    x_size = cuda.blockDim.x * cuda.gridDim.x
    y_size = cuda.blockDim.y * cuda.gridDim.y
    
    #get the range of x and y
    range_x = (width - 1) // (x_size + 1)
    range_y = (height - 1) // (y_size + 1)
    
    #calculate the mandel value for each element of the block as the original program
    for x_index in range(range_x):
        new_x = x_size * x_index + x0
        real = min_x + new_x * pixel_size_x
        for y_index in range(range_y):
            new_y = y_size * y_index + y0
            imag = min_y + new_y * pixel_size_y
            if (new_x < width and new_y < height):
                image[new_y, new_x] = mandel(real, imag, iters)

            
if __name__ == '__main__':
    image = np.zeros((1024, 1536), dtype = np.uint8)
    blockdim = (32, 8)
    griddim = (32, 16)

    image_global_mem = cuda.to_device(image)
    compute_mandel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20) 
    image_global_mem.copy_to_host()
    imshow(image)
    show()


