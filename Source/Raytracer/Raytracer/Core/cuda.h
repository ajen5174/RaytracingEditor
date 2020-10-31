#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CheckCudaErrors(val) CheckCuda( (val), #val, __FILE__, __LINE__ )
void CheckCuda(cudaError_t result, char const* const func, const char* const file, int const line);