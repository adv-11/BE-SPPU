#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

const char* kernelSource = 
"__kernel void print_name_month(__global const char* name, __global const char* month) {\n"
"   int gid = get_global_id(0);\n"
"   printf(\"Global ID: %d - Name: %s, Month: %s\\n\", gid, name, month);\n"
"}\n";

int main() {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    // Setup
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    // Compile kernel
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "print_name_month", &err);

    // Data
    const char* name = "Advait";
    const char* month = "April";
    cl_mem name_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 20, (void*)name, &err);
    cl_mem month_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 20, (void*)month, &err);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &name_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &month_buf);

    // Run kernel
    size_t globalSize = 5;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    clFinish(queue);

    // Cleanup
    clReleaseMemObject(name_buf);
    clReleaseMemObject(month_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
