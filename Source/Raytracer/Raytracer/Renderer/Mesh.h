#pragma once
#include <vector>
#include "../Math/Triangle.h"
#include "../Math/vec3.h"



class Mesh
{
public:
	__device__ Mesh()
	{
		
	
		


	}

public:
	Triangle* vertices;
	int numTriangles;
};