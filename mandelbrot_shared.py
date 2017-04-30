

# 
# A CUDA version to calculate the Mandelbrot set
#
from numba import cuda, float32, int32
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
    # Define the arrays in the shared memory
    
    #first, as the initial program, calculate pixel_size_x and pixel_size_y
    Sheight = cuda.shared.array(shape=(1), dtype=int32)
    Swidth = cuda.shared.array(shape=(1), dtype=int32)
    Sheight[0] = image.shape[0]
    Swidth[0] = image.shape[1]
    
    Spixel_size_x = cuda.shared.array(shape=(1), dtype=float32)
    Spixel_size_y = cuda.shared.array(shape=(1), dtype=float32)
    Spixel_size_x[0] = (max_x - min_x) / Swidth[0]
    Spixel_size_y[0] = (max_y - min_y) / Sheight[0]
    
    #as required, obtain the starting x and y coordinates using cuda.grid()
    x0, y0=cuda.grid(2)
    
    #obtain the size of the block using gridDim and blockDim
    Sx_size = cuda.shared.array(shape=(1), dtype=int32)
    Sy_size = cuda.shared.array(shape=(1), dtype=int32)
    Sx_size[0] = cuda.blockDim.x * cuda.gridDim.x
    Sy_size[0] = cuda.blockDim.y * cuda.gridDim.y
    
    #get the range of x and y
    Srange_x = cuda.shared.array(shape=(1), dtype=int32)
    Srange_y = cuda.shared.array(shape=(1), dtype=int32)
    Srange_x[0] = (Swidth[0] - 1) // (Sx_size[0] + 1)
    Srange_y[0] = (Sheight[0] - 1) // (Sy_size[0] + 1)
    
    # Wait until all threads finish preloading
    cuda.syncthreads()

        
    #calculate the mandel value for each element of the block as the original program
    for x_index in range(Srange_x[0]):
        new_x = Sx_size[0] * x_index + x0
        real = min_x + new_x * Spixel_size_x[0]
        for y_index in range(Srange_y[0]):
            new_y = Sy_size[0] * y_index + y0
            imag = min_y + new_y * Spixel_size_y[0]
            if (new_x < Swidth[0] and new_y < Sheight[0]):
                image[new_y, new_x] = mandel(real, imag, iters)
                
                # Wait until all threads finish computing
                cuda.syncthreads()

            
if __name__ == '__main__':
    image = np.zeros((1024, 1536), dtype = np.uint8)
    blockdim = (32, 8)
    griddim = (32, 16)

    image_global_mem = cuda.to_device(image)
    compute_mandel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20) 
    image_global_mem.copy_to_host()
    imshow(image)
    show()




