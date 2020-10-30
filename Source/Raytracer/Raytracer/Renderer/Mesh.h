#pragma once
#include <vector>
#include "../Math/Triangle.h"
#include "../Math/vec3.h"

#include "../Raytracer.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>




class Mesh
{
public:
    //this is a device only call, because we need the space allocated in device memory for our members, not host
	__device__ Mesh()
	{
		
        boundingSphere = new Sphere(vec3(0.0f), 0.0f);
	}

public:
    vec3* vertices;
	Triangle** triangles;
	int numTriangles;
    //temporary render speed up, surrounds all triangles in the mesh
    Sphere* boundingSphere;
};
//this method load in whatever data we need from disk and slaps it into device accessible memory.
inline void CreateMesh(std::string path, Mesh* mesh)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals);

    int numVertices = scene->mMeshes[0]->mNumVertices;
    //mesh->vertices = new vec3[numVertices];
    CheckCudaErrors(cudaMallocManaged(&mesh->vertices, (numVertices) * sizeof(vec3)));

    for (int i = 0; i < scene->mMeshes[0]->mNumVertices; i++)
    {
        aiVector3D temp = scene->mMeshes[0]->mVertices[i];
        mesh->vertices[i].x = temp.x;
        mesh->vertices[i].y = temp.y;
        mesh->vertices[i].z = temp.z;
    }

    CheckCudaErrors(cudaMallocManaged((void**)&(mesh->triangles), (numVertices / 3) * sizeof(Triangle*)));

    mesh->numTriangles = numVertices / 3;
    
}