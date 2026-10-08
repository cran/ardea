## ----setup_chunk_opts, include = FALSE----------------------------------------
knitr::opts_chunk$set(results = 'hold')

## ----intro--------------------------------------------------------------------
library(ardea)

opencl_is_available()
cuda_is_available()
metal_is_available()

## ----device_funs--------------------------------------------------------------
# write out a character vector to a tempfile for opencl, metal, and cuda
opencl_mm_naive <- '
__kernel void opencl_mm_naive(__global float* output,
                              const long M,
                              const long K,
                              const long N,
                              __global const float* A,
                              __global const float* B)
{
    long row = get_global_id(0);
    long col = get_global_id(1);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (long k = 0; k < K; k++) {
            sum += A[row + k * M] * B[k + col * K];
        }
        output[row + col * M] = sum;
    }
}
'
tmp01 <- tempfile(fileext = ".cl")
writeLines(text = opencl_mm_naive,
           con = tmp01)

cuda_mm_naive <- '
extern "C" __global__ void cuda_mm_naive(float* output,
                                         const long long M,
                                         const long long K,
                                         const long long N,
                                         const float* A,
                                         const float* B)
{
    long long row = blockIdx.x * blockDim.x + threadIdx.x;
    long long col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (long long k = 0; k < K; k++) {
            sum += A[row + k * M] * B[k + col * K];
        }
        output[row + col * M] = sum;
    }
}
'
tmp02 <- tempfile(fileext = ".cu")
tmp03 <- tempfile(fileext = ".ptx")
writeLines(text = cuda_mm_naive,
           con = tmp02)

metal_mm_naive <- '
kernel void metal_mm_naive(device float* output [[buffer(0)]],
                           constant uint& M [[buffer(1)]],
                           constant uint& K [[buffer(2)]],
                           constant uint& N [[buffer(3)]],
                           device const float* A [[buffer(4)]],
                           device const float* B [[buffer(5)]],
                           uint2 id [[thread_position_in_grid]])
{
    uint row = id.x;
    uint col = id.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (uint k = 0; k < K; k++) {
            sum += A[row + k * M] * B[k + col * K];
        }
        output[row + col * M] = sum;
    }
}
'
tmp04 <- tempfile(fileext = ".metal")
tmp05 <- tempfile(fileext = ".metallib")
writeLines(text = metal_mm_naive,
           con = tmp04)

## ----setup--------------------------------------------------------------------
# generic inputs, regardless of framework
var00 <- vector(mode = "numeric",
               length = 250000)
var01 <- matrix(rnorm(250000),
               nrow = 500,
               ncol = 500)
var02 <- matrix(rnorm(250000),
               nrow = 500,
               ncol = 500)
dim01 <- nrow(var01)
dim02 <- ncol(var01)
# dim03 <- nrow(var02)
dim04 <- ncol(var02)

arg_list <- list(var00, # our expected output vector, we've just filled it with zeros
                 dim01,
                 dim02,
                 dim04,
                 as.vector(var01),
                 as.vector(var02))

## ----typing_diffs-------------------------------------------------------------
opencl_arg_types <- c("float",
                      "long",
                      "long",
                      "long",
                      "float",
                      "float")
metal_arg_types = c("float",
                    "uint",
                    "uint",
                    "uint",
                    "float",
                    "float")
cuda_arg_types <- c("float",
                    "long",
                    "long",
                    "long",
                    "float",
                    "float")

## ----prepping_dispatch--------------------------------------------------------
if (opencl_is_available() &
    opencl_devices_exist()) {
  print("OpenCL is available on this system!")
  cl_dvcs <- opencl_device_information()
  cl_ctx <- opencl_make_context(device = cl_dvcs[[1]])
  cl_program <- opencl_make_program(cl_file = tmp01,
                                    context = cl_ctx)
  cl_knl <- opencl_make_kernelptr(program = cl_program,
                                  kernel_names = "opencl_mm_naive")
}
if (cuda_is_available() &
    cuda_devices_exist()) {
  print("CUDA is available on this system!")
  cu_dvcs <- cuda_device_information()
  cu_ctx <- cuda_make_context(device = cu_dvcs[[1]])
  cu_program <- cuda_make_program(cuda_file = tmp02,
                                  context = cu_ctx,
                                  ptx_file = tmp03)
  cu_knl <- cuda_make_kernelptr(program = cu_program,
                                context = cu_ctx,
                                kernel_names = "cuda_mm_naive")
}
if (metal_is_available() &
    metal_devices_exist()) {
  print("Metal is available on this system!")
  mtl_dvcs <- metal_device_information()
  mtl_ctx <- metal_make_context(device = mtl_dvcs[[1]])
  mtl_program <- metal_make_program(metal_file = tmp04,
                                    metallib_file = tmp05,
                                    context = mtl_ctx)
  mtl_knl <- metal_make_kernelptr(program = mtl_program,
                                  context = mtl_ctx,
                                  kernel_names = "metal_mm_naive")
}

## ----execute_dispatch---------------------------------------------------------
# our builtin optimized CPU implementation
system.time(res01 <- var01 %*% var02) 
if (opencl_is_available() &
    opencl_devices_exist()) {
  print("OpenCL implementation:")
  system.time(res02 <- simple_wrapper(framework = "opencl",
                                      context_ptr = cl_ctx,
                                      kernel_ptr = cl_knl$opencl_mm_naive,
                                      arg_types = opencl_arg_types,
                                      arg_list = arg_list,
                                      problem_dims = as.integer(c(dim01,
                                                                  dim04,
                                                                  1L)),
                                      group_dims = NULL,
                                      workers_per = NULL))
  plot(as.vector(res01),
       res02,
       pch = 46)
}
if (cuda_is_available() &
    cuda_devices_exist()) {
  print("CUDA implementation:")
  system.time(res03 <- simple_wrapper(framework = "cuda",
                                      context_ptr = cu_ctx,
                                      kernel_ptr = cu_knl$cuda_mm_naive,
                                      arg_types = cuda_arg_types,
                                      arg_list = arg_list,
                                      problem_dims = as.integer(c(dim01,
                                                                  dim04,
                                                                  1L)),
                                      group_dims = NULL,
                                      workers_per = NULL))
  plot(as.vector(res01),
       res03,
       pch = 46)
}
if (metal_is_available() &
    metal_devices_exist()) {
  print("Metal implementation:")
  system.time(res03 <- simple_wrapper(framework = "metal",
                                      context_ptr = mtl_ctx,
                                      kernel_ptr = mtl_knl$metal_mm_naive,
                                      arg_types = metal_arg_types,
                                      arg_list = arg_list,
                                      problem_dims = as.integer(c(dim01,
                                                                  dim04,
                                                                  1L)),
                                      group_dims = NULL,
                                      workers_per = NULL))
  plot(as.vector(res01),
       res03,
       pch = 46)
}

## -----------------------------------------------------------------------------
sessionInfo()

