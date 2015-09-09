/// @copyright (c) 2007 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia
///
/// This file is part of the ASKAP software distribution.
///
/// The ASKAP software distribution is free software: you can redistribute it
/// and/or modify it under the terms of the GNU General Public License as
/// published by the Free Software Foundation; either version 2 of the License,
/// or (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program; if not, write to the Free Software
/// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
///
/// This program was modified so as to use it in the contest.
/// The last modification was on April 2, 2015.
///

// Include own header file first
#include "Benchmark.h"

// System includes
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
/*
cl_program load_program(cl_context context, const char* filename)
{
    std::ifstream in(filename, std::ios_base::binary);
    if(!in.good()) {
        return 0;
    }

    // get file length
    in.seekg(0, std::ios_base::end);
    size_t length = in.tellg();
    in.seekg(0, std::ios_base::beg);

    // read program source
    std::vector<char> data(length + 1);
    in.read(&data[0], length);
    data[length] = 0;

    // create and build program 
    const char* source = &data[0];
    cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
        if(program == 0) {
        return 0;
    }

    if(clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS) {
        return 0;
    }

    return program;
}
*/
cl_program load_program(cl_context context, cl_device_id device, const char* filename)
{
    FILE *fp = fopen(filename, "rt");
    size_t length;
    char *data;
    char *build_log;
    size_t ret_val_size;
    cl_program program = 0;
    cl_int status = 0;

    if(!fp) return 0;

    // get file length
    fseek(fp, 0, SEEK_END);
    length = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // read program source
    data = (char *)malloc(length + 1);
    fread(data, sizeof(char), length, fp);
    data[length] = '\0';

    // create and build program 
    program = clCreateProgramWithSource(context, 1, (const char **)&data, 0, 0);
    if (program == 0) return 0;

    status = clBuildProgram(program, 0, 0, 0, 0, 0);
    if (status != CL_SUCCESS) {
        printf("Error:  Building Program from file %s\n", filename);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
        build_log = (char *)malloc(ret_val_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
        build_log[ret_val_size] = '\0';
        printf("Building Log:\n%s", build_log);
        return 0;
    }

    return program;
}

Benchmark::Benchmark()
        : next(1)
{
}

// Return a pseudo-random integer in the range 0..2147483647
// Based on an algorithm in Kernighan & Ritchie, "The C Programming Language"
int Benchmark::randomInt()
{
    const unsigned int maxint = std::numeric_limits<int>::max();
    next = next * 1103515245 + 12345;
    return ((unsigned int)(next / 65536) % maxint);
}

void Benchmark::init()
{
    int rd1_tag,rd2_tag,rd3_tag,rd4_tag,id,nl;

    rd1_tag=0;
    rd2_tag=1;
    rd3_tag=2;
    rd4_tag=3;

    nSamples=nSamples_a;
    nl=nSamples;
    rd1 = new Coord[nl];
    rd2 = new Coord[nl];
    rd3 = new Coord[nl];
    rd4 = new Coord[nl];
    
    // Initialize the data to be gridded
    u.resize(nSamples);
    v.resize(nSamples);
    w.resize(nSamples);
    samples.resize(nSamples*nChan);   

	Coord rd;
	FILE * fp;
	if( (fp=fopen("randnum.dat","rb"))==NULL )
	{
		printf("cannot open file\n");
		return;
	}
     
	for (int i = 0; i < nSamples; i++) {
		if(fread(&rd,sizeof(Coord),1,fp)!=1){printf("Rand number read error!\n");}
		rd1[i]=rd;
		if(fread(&rd,sizeof(Coord),1,fp)!=1){printf("Rand number read error!\n");}
		rd2[i]=rd;
		if(fread(&rd,sizeof(Coord),1,fp)!=1){printf("Rand number read error!\n");}
		rd3[i]=rd;
		if(fread(&rd,sizeof(Coord),1,fp)!=1){printf("Rand number read error!\n");}
		rd4[i]=rd;
	}

	fclose(fp);

    for (int i = 0; i < nSamples; i++) {
      u[i] = baseline * rd1[i] - baseline / 2;
      v[i] = baseline * rd2[i] - baseline / 2;
      w[i] = baseline * rd3[i] - baseline / 2;
      for (int chan = 0; chan < nChan; chan++) {
        Coord c2=Coord(chan)/Coord(nChan);
        samples[i*nChan+chan].data=Value(rd4[i]+c2,rd4[i]-c2);
      }
    }

    grid =new Value[gSize*gSize];
    for (int i = 0; i < gSize*gSize; i++){
      grid[i]=Value(0.0);
    }

    // Measure frequency in inverse wavelengths
    std::vector<Coord> freq(nChan);

    for (int i = 0; i < nChan; i++) {
        freq[i] = (1.4e9 - 2.0e5 * Coord(i) / Coord(nChan)) / 2.998e8;
    }

    // Initialize convolution function and offsets
    initC(freq, cellSize, wSize, m_support, overSample, wCellSize, C);
    initCOffset(u, v, w, freq, cellSize, wCellSize, wSize, gSize,
                m_support, overSample);

    delete [] rd1;
    delete [] rd2;
    delete [] rd3;
    delete [] rd4;
}

void Benchmark::runGrid()
{
    gridKernel(m_support, C, grid, gSize);
}

/////////////////////////////////////////////////////////////////////////////////
// The next function is the kernel of the gridding.
// The data are presented as a vector. Offsets for the convolution function
// and for the grid location are precalculated so that the kernel does
// not need to know anything about world coordinates or the shape of
// the convolution function. The ordering of cOffset and iu, iv is
// random.
//
// Perform gridding
//
// data - values to be gridded in a 1D vector
// support - Total width of convolution function=2*support+1
// C - convolution function shape: (2*support+1, 2*support+1, *)
// cOffset - offset into convolution function per data point
// iu, iv - integer locations of grid points
// grid - Output grid: shape (gSize, *)
// gSize - size of one axis of grid
void Benchmark::gridKernel(const int support,
                           const std::vector<Value>& C,
                           Value * grid, const int gSize)
{
    cl_int err;
    cl_uint num;
    err = clGetPlatformIDs(0, 0, &num);
    if(err != CL_SUCCESS) {
        std::cerr << "Unable to get platforms\n";
        exit(1);
    }   

    std::vector<cl_platform_id> platforms(num);
    err = clGetPlatformIDs(num, &platforms[0], &num);
    if(err != CL_SUCCESS) {
        std::cerr << "Unable to get platform ID\n";
        exit(1);
    }

    cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0 };
    cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
    if(context == 0) {
        std::cerr << "Can't create OpenCL context\n";
        exit(1);
    }

    size_t cb;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    std::vector<cl_device_id> devices(cb / sizeof(cl_device_id));
    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);

    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
    std::string devname;
    devname.resize(cb);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
    std::cout << "Device: " << devname.c_str() << "\n";

    clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &cb);
    std::vector<size_t> max_work_item_sizes(cb / sizeof(size_t));
    clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, cb, &max_work_item_sizes[0], 0);

    std::cout << "Max work item sizes " << max_work_item_sizes[0] << "  " << max_work_item_sizes[1] << "  " << max_work_item_sizes[2] << "\n";

    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, 0);
    if(queue == 0) {
        std::cerr << "Can't create command queue\n";
        clReleaseContext(context);
        exit(1);
    }
    
    std::cout << "original: " << samples.size() << "  " << sSize << "\n";
    for (int dind = 0; dind < int(samples.size()); ++dind) {
        // The actual grid point from which we offset
        int gind = samples[dind].iu + gSize * samples[dind].iv - support;

        // The Convoluton function point from which we offset
        int cind = samples[dind].cOffset;

        for (int suppv = 0; suppv < sSize; suppv++) {
            Value* gptr = &grid[gind];
            const Value* cptr = &C[cind];
            const Value d = samples[dind].data;
            for (int suppu = 0; suppu < sSize; suppu++) {
                *(gptr++) += d * (*(cptr++));
                //std::cout << "Iteration: " << dind << "  " << suppv << "  " << suppu << "\n";
            }

            gind += gSize;
            cind += sSize;
        }
    }

    for (int i = 0; i < gSize * gSize; i++) {
        printf("grid.real %d\n", grid[i].real());
        printf("grid.imag %d\n", grid[i].imag());
    } 

    // deal with samples data
    std::vector<int> iu(int(samples.size())), iv(int(samples.size())), cOffset(int(samples.size()));
    std::vector<double> data_real(int(samples.size())), data_imag(int(samples.size()));
    for (int i = 0; i < int(samples.size()); i++) {
        iu[i] = samples[i].iu;
        iv[i] = samples[i].iv;
        cOffset[i] = samples[i].cOffset;
        data_real[i] = samples[i].data.real();
        data_imag[i] = samples[i].data.imag();
    }
    cl_mem cl_iu = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
            sizeof(cl_int) * samples.size(), &iu[0], NULL);
    cl_mem cl_iv = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
            sizeof(cl_int) * samples.size(), &iv[0], NULL);
    cl_mem cl_cOffset = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
            sizeof(cl_int) * samples.size(), &cOffset[0], NULL);
    cl_mem cl_data_real = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
            sizeof(cl_double) * samples.size(), &data_real[0], NULL);
    cl_mem cl_data_imag = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_double) * samples.size(), &data_imag[0], NULL);

    std::cout << "gSize: " << gSize << "  " << "support" << support << "\n";
    // deal with gSize, sSize and support
    int gSize_temp = gSize;
    int support_temp = support;
    int sSize_temp = sSize;
    cl_mem cl_gSize = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(int), &gSize_temp, NULL);
    cl_mem cl_support = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(int), &support_temp, NULL);


    // deal with grid data
    std::vector<double> grid_real(gSize * gSize), grid_imag(gSize *gSize);
    for (int i = 0; i < gSize * gSize; i++) {
        grid_real[i] = grid[i].real();
        grid_imag[i] = grid[i].imag();
    }

    cl_mem cl_grid_real = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_double) * (gSize * gSize), &grid_real[0], NULL);
    cl_mem cl_grid_imag = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_double) *(gSize * gSize), &grid_imag[0], NULL);
    
    // deal with C data
    std::vector<double> C_real(C.size()), C_imag(C.size());
    for (int i = 0; i < C.size(); i++) {
        C_real[i] = C[i].real();
        C_imag[i] = C[i].imag();
    }

    cl_mem cl_C_real = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_double) * C.size(), &C_real[0], NULL);
    cl_mem cl_C_imag = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_double) * C.size(), &C_imag[0], NULL);

    if (cl_iu == 0 || cl_iv == 0 || cl_cOffset == 0 || cl_data_real == 0 || cl_data_imag == 0 ||
            cl_grid_real == 0 || cl_grid_imag == 0 || cl_C_real == 0 || cl_C_imag == 0) {
        std::cerr << "Can't create OpenCL buffer\n";
        clReleaseMemObject(cl_iu);
        clReleaseMemObject(cl_iv);
        clReleaseMemObject(cl_cOffset);
        clReleaseMemObject(cl_data_real);
        clReleaseMemObject(cl_data_imag);
        clReleaseMemObject(cl_grid_real);
        clReleaseMemObject(cl_grid_imag);
        clReleaseMemObject(cl_C_real);
        clReleaseMemObject(cl_C_imag);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(1);
    }

    cl_program program = load_program(context, devices[0], "gridKernel.cl");
    // cl_program program = load_program(context, "adder.cl");
    if (program == 0) {
        std::cerr << "Can't load or build program\n";
        clReleaseMemObject(cl_iu);
        clReleaseMemObject(cl_iv);
        clReleaseMemObject(cl_cOffset);
        clReleaseMemObject(cl_data_real);
        clReleaseMemObject(cl_data_imag);
        clReleaseMemObject(cl_grid_real);
        clReleaseMemObject(cl_grid_imag);
        clReleaseMemObject(cl_C_real);
        clReleaseMemObject(cl_C_imag);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(1);   
    }

    cl_kernel gridKernel = clCreateKernel(program, "grid", 0);
    if (grid == 0) {
        std::cerr << "Can't load kernel\n";
        clReleaseMemObject(cl_iu);
        clReleaseMemObject(cl_iv);
        clReleaseMemObject(cl_cOffset);
        clReleaseMemObject(cl_data_real);
        clReleaseMemObject(cl_data_imag);
        clReleaseMemObject(cl_grid_real);
        clReleaseMemObject(cl_grid_imag);
        clReleaseMemObject(cl_C_real);
        clReleaseMemObject(cl_C_imag);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(1);
    }
    

    clSetKernelArg(gridKernel, 0, sizeof(cl_mem), &cl_iu);
    clSetKernelArg(gridKernel, 1, sizeof(cl_mem), &cl_iv);
    clSetKernelArg(gridKernel, 2, sizeof(cl_uint), &gSize_temp);
    clSetKernelArg(gridKernel, 3, sizeof(cl_uint), &sSize_temp);
    clSetKernelArg(gridKernel, 4, sizeof(cl_uint), &support_temp);
    clSetKernelArg(gridKernel, 5, sizeof(cl_mem), &cl_cOffset);
    clSetKernelArg(gridKernel, 6, sizeof(cl_mem), &cl_grid_real);
    clSetKernelArg(gridKernel, 7, sizeof(cl_mem), &cl_grid_imag);
    clSetKernelArg(gridKernel, 8, sizeof(cl_mem), &cl_C_real);
    clSetKernelArg(gridKernel, 9, sizeof(cl_mem), &cl_C_imag);
    clSetKernelArg(gridKernel, 10, sizeof(cl_mem), &cl_data_real);
    clSetKernelArg(gridKernel, 11, sizeof(cl_mem), &cl_data_imag);

    size_t global_work_size[] = { sSize, sSize, samples.size()};
    std::cout << "sSize: " << sSize << "  " << "gSize: " << gSize;
    err = clEnqueueNDRangeKernel(queue, gridKernel, 3, 0, global_work_size, 0, 0, 0, 0);
    std::vector<double> res_grid_real(gSize * gSize), res_grid_imag(gSize * gSize);
    if (err == CL_SUCCESS) { 
        std::cout << "SUCCESS"; 
        err = clEnqueueReadBuffer(queue, cl_grid_real, CL_TRUE, 0, sizeof(double) * gSize * gSize, &res_grid_real[0], 0, 0, 0);
    }

    
    clReleaseMemObject(cl_iu);
    clReleaseMemObject(cl_iv);
    clReleaseMemObject(cl_cOffset);
    clReleaseMemObject(cl_data_real);
    clReleaseMemObject(cl_data_imag);
    clReleaseMemObject(cl_grid_real);
    clReleaseMemObject(cl_grid_imag);
    clReleaseMemObject(cl_C_real);
    clReleaseMemObject(cl_C_imag);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

/////////////////////////////////////////////////////////////////////////////////
// Initialize W project convolution function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// wSize - Size of lookup table in w
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
void Benchmark::initC(const std::vector<Coord>& freq,
                      const Coord cellSize, const int wSize,
                      int& support, int& overSample,
                      Coord& wCellSize, std::vector<Value>& C)
{
    support = static_cast<int>(1.5 * sqrt(std::abs(baseline) * static_cast<Coord>(cellSize)
                                          * freq[0]) / cellSize);

    overSample = 8;
    wCellSize = 2 * baseline * freq[0] / wSize;

    // Convolution function. This should be the convolution of the
    // w projection kernel (the Fresnel term) with the convolution
    // function used in the standard case. The latter is needed to
    // suppress aliasing. In practice, we calculate entire function
    // by Fourier transformation. Here we take an approximation that
    // is good enough.
    sSize = 2 * support + 1;

    const int cCenter = (sSize - 1) / 2;

    C.resize(sSize*sSize*overSample*overSample*wSize);

    double rr,ri;
    for (int k = 0; k < wSize; k++) {
        double w = double(k - wSize / 2);
        double fScale = sqrt(std::abs(w) * wCellSize * freq[0]) / cellSize;

        for (int osj = 0; osj < overSample; osj++) {
            for (int osi = 0; osi < overSample; osi++) {
                for (int j = 0; j < sSize; j++) {
                    double j2 = std::pow((double(j - cCenter) + double(osj) / double(overSample)), 2);

                    for (int i = 0; i < sSize; i++) {
                        double i2 = std::pow((double(i - cCenter) + double(osi) / double(overSample)), 2);
                        double r2 = j2 + i2 + sqrt(j2*i2);
                        long int cind = i + sSize * (j + sSize * (osi + overSample * (osj + overSample * k)));

                        if (w != 0.0) {
                            rr=std::cos(r2 / (w * fScale));
                            ri=std::sin(r2 / (w * fScale));
                            C[cind] = static_cast<Value>(rr,ri);
                        } else {
                            rr=std::exp(-r2);
                            C[cind] = static_cast<Value>(rr);
                        }
                    }
                }
            }
        }
    }

    // Now normalise the convolution function
    Coord sumC = 0.0;

    for (int i = 0; i < sSize*sSize*overSample*overSample*wSize; i++) {
        sumC += std::abs(C[i]);
    }

    for (int i = 0; i < sSize*sSize*overSample*overSample*wSize; i++) {
        C[i] *= Value(wSize * overSample * overSample / sumC);
    }
}

// Initialize Lookup function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// gSize - size of grid in pixels (per axis)
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
// wSize - Size of lookup table in w
void Benchmark::initCOffset(const std::vector<Coord>& u, const std::vector<Coord>& v,
                            const std::vector<Coord>& w, const std::vector<Coord>& freq,
                            const Coord cellSize, const Coord wCellSize,
                            const int wSize, const int gSize, const int support,
                            const int overSample)
{
    const int nSamples = u.size();
    const int nChan = freq.size();

    // Now calculate the offset for each visibility point
    for (int i = 0; i < nSamples; i++) {
        for (int chan = 0; chan < nChan; chan++) {

            int dind = i * nChan + chan;

            Coord uScaled = freq[chan] * u[i] / cellSize;
            samples[dind].iu = int(uScaled);

            if (uScaled < Coord(samples[dind].iu)) {
                samples[dind].iu -= 1;
            }

            int fracu = int(overSample * (uScaled - Coord(samples[dind].iu)));
            samples[dind].iu += gSize / 2;

            Coord vScaled = freq[chan] * v[i] / cellSize;
            samples[dind].iv = int(vScaled);

            if (vScaled < Coord(samples[dind].iv)) {
                samples[dind].iv -= 1;
            }

            int fracv = int(overSample * (vScaled - Coord(samples[dind].iv)));
            samples[dind].iv += gSize / 2;

            // The beginning of the convolution function for this point
            Coord wScaled = freq[chan] * w[i] / wCellSize;
            int woff = wSize / 2 + int(wScaled);
            samples[dind].cOffset = sSize * sSize * (fracu + overSample * (fracv + overSample * woff));
        }
    }
}

void Benchmark::printGrid()
{
  FILE * fp;
  if( (fp=fopen("grid.dat","wb"))==NULL )
  {
    printf("cannot open file\n");
    return;
  }  

  unsigned ij;
  for (int i = 0; i < gSize; i++)
  {
    for (int j = 0; j < gSize; j++)
    {
      ij=j+i*gSize;
      if(fwrite(&grid[ij],sizeof(Value),1,fp)!=1)
        printf("File write error!\n"); 

    }
  }
  
  fclose(fp);
}

int Benchmark::getsSize()
{
    return sSize;
}

int Benchmark::getSupport()
{
    return m_support;
};
