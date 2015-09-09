#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif


__kernel void grid(__global const int* iu, __global const int* iv, const int gSize, const int sSize, const int support,
                            __global const int* cOffset, __global double* grid_real, __global double* grid_imag,
                            __global const double* C_real, __global const double* C_imag, __global const double* data_real,
                            __global const double* data_imag)
{
    int suppu = get_global_id(0);
    int suppv = get_global_id(1);
    int dind = get_global_id(2);

    int gind = iu[dind] + gSize * iv[dind] - support;
    int cind = cOffset[dind];

    double cptr_real = C_real[(cind + (suppv * sSize)) + suppu];
    double cptr_imag = C_imag[(cind + (suppv * sSize)) + suppu];

    double d_real = data_real[dind];
    double d_imag = data_imag[dind];

    grid_real[(gind + (suppv * gSize)) + suppu] = grid_real[(gind + (suppv * gSize)) + suppu] +
        d_real * cptr_real;
    grid_imag[(gind + (suppv * gSize)) + suppu] = grid_imag[(gind + (suppv * gSize)) + suppu] +
        d_imag * cptr_imag;

    // printf("gSize %d\n", gSize);
    // printf("sSize %d\n", sSize);
    // printf("support %d\n", support);

    // printf("GPU %d,%d,%d\n", suppu, suppv, dind);
}
