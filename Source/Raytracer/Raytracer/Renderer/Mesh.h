#pragma once
#include <vector>
#include "../Math/Triangle.h"
#include "../Math/vec3.h"
#include "../Math/Sphere.h"
#include "../Core/Transform.h"
#include "../Core/cuda.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "../Core/Json.h"
#include "Material.h"
#include "../Math/BVHNode.h"


inline __global__ void CreateMeshTriangles(Triangle** triangles, int numTriangles, vec3* vertices, int numVertices, Material* material, Transform* transform)
{
	//transform the vertices lmao
	mat4 t = transform->GetMatrix();

	for (int i = 0; i < numTriangles * 3; i++)
	{
		vertices[i] = t * vertices[i];
	}

	for (int i = 0; i < numTriangles; i++)
	{
		//if (i % 3 == 0)
		{
			triangles[i] = new Triangle(vertices[(i * 3) + 0], vertices[(i * 3) + 1], vertices[(i * 3) + 2]);
			triangles[i]->material = material;
		}
	}

	//vec3 averagePosition;
	//averagePosition = vec3();
	//for (int i = 0; i < numTriangles * 3; i++)
	//{
	//	averagePosition = averagePosition + vertices[i];
	//}
	//averagePosition = averagePosition / (numTriangles * 3);
	//float boundingRadius = 0.0f;
	//for (int i = 0; i < numTriangles * 3; i++)
	//{
	//	float tempRadius = (averagePosition - mesh->vertices[i]).Magnitude();
	//	if (tempRadius > boundingRadius)
	//		boundingRadius = tempRadius;
	//}
	//mesh->boundingSphere = new Sphere(averagePosition, boundingRadius);

}


class Mesh
	: public Hittable
{
public:
    //this is a device only call, because we need the space allocated in device memory for our members, not host
	__device__ Mesh()
	{
		
        //boundingSphere = new Sphere(vec3(0.0f), 0.0f);
	}

	__host__ __device__ virtual bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) const override
	{
		bool hit = false;
		float closestSoFar = maxDist;
		HitInfo tempInfo;
		if (box.Hit(ray, minDist, maxDist))
		{
			for (int i = 0; i < numTriangles; i++)
			{
				if (triangles[i]->Hit(ray, 0.001f, closestSoFar, tempInfo))
				{
				    hitInfo = tempInfo;
				    closestSoFar = hitInfo.distance;
				    hit = true;

				}
			}
		}
		return hit;
	}
	__host__ __device__ virtual bool BoundingBox(AABB& outputBox) override
	{

		vec3 min(100000000.0f);
		vec3 max(-100000000.0f);

		for (int i = 0; i < numTriangles * 3; i++)
		{
			min.x = vertices[i].x < min.x ? vertices[i].x : min.x;
			min.y = vertices[i].y < min.y ? vertices[i].y : min.y;
			min.z = vertices[i].z < min.z ? vertices[i].z : min.z;

			max.x = vertices[i].x > max.x ? vertices[i].x : max.x;
			max.y = vertices[i].y > max.y ? vertices[i].y : max.y;
			max.z = vertices[i].z > max.z ? vertices[i].z : max.z;
		}
		outputBox = AABB(min, max);
		return true;
	}

	bool Load(rapidjson::Value& value)
	{
		std::string path;
		json::GetString(value, "meshPath", path);
		LoadFromFile(path);
		return true;
	}

	//this method load in whatever data we need from disk and slaps it into device accessible memory.
	void LoadFromFile(std::string path)
	{
		Assimp::Importer importer;
		const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals);

		int numVertices = scene->mMeshes[0]->mNumVertices;
		//mesh->vertices = new vec3[numVertices];
		CheckCudaErrors(cudaMallocManaged(&vertices, (numVertices) * sizeof(vec3)));

		for (int i = 0; i < scene->mMeshes[0]->mNumVertices; i++)
		{
			aiVector3D temp = scene->mMeshes[0]->mVertices[i];
			vertices[i].x = temp.x;
			vertices[i].y = temp.y;
			vertices[i].z = temp.z;
		}

		CheckCudaErrors(cudaMallocManaged((void**)&(triangles), (numVertices / 3) * sizeof(Triangle*)));
		CheckCudaErrors(cudaDeviceSynchronize());

		numTriangles = numVertices / 3;

		CreateMeshTriangles << <1, 1 >> > (triangles, numTriangles, vertices, numVertices, material, parentTransform);
		CheckCudaErrors(cudaDeviceSynchronize());

		//instead of allocating the triangles on the GPU we allocate the BVHNode on the GPU
		//CheckCudaErrors(cudaMallocManaged((void**)&(bvh), sizeof(BVHNode*)));
		//clock_t start, stop;
		//start = clock();
		//std::cout << "Creating BVH\n";
		//CheckCudaErrors(cudaDeviceSynchronize()); 
		//CreateBVHNode<<<1, 1>>>(bvh, triangles, numTriangles);
		//CheckCudaErrors(cudaDeviceSynchronize());
		//stop = clock();
		//double seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
		//std::cout << "BVH Complete in " << seconds << " seconds.\n";

		//make box because we have to "new" on GPU
		vec3 min(100000000.0f);
		vec3 max(-100000000.0f);

		for (int i = 0; i < numTriangles * 3; i++)
		{
			min.x = vertices[i].x < min.x ? vertices[i].x : min.x;
			min.y = vertices[i].y < min.y ? vertices[i].y : min.y;
			min.z = vertices[i].z < min.z ? vertices[i].z : min.z;

			max.x = vertices[i].x > max.x ? vertices[i].x : max.x;
			max.y = vertices[i].y > max.y ? vertices[i].y : max.y;
			max.z = vertices[i].z > max.z ? vertices[i].z : max.z;
		}
		box = AABB((min), (max));

		//std::vector<Hittable*> triangleList;
		//for (int i = 0; i < numTriangles; i++)
		//{
		//	triangleList.push_back((Hittable*)triangles[i]);
		//}
		//
		//bvh->CreateTree(triangleList, 0, numTriangles);

	}

public:
    vec3* vertices;
	Triangle** triangles;
	int numTriangles;
	//BVHNode** bvh;
	AABB box;
	Transform* parentTransform;
    //temporary render speed up, surrounds all triangles in the mesh
    //Sphere* boundingSphere;
	Material* material;
};
