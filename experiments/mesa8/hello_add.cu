// Mesa 8: Hello World Kernel
// The simplest possible CUDA kernel: add two integers

__global__ void add_kernel(int *a, int *b, int *c) {
    *c = *a + *b;
}

// Vector add for more interesting SASS
__global__ void vadd_kernel(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
