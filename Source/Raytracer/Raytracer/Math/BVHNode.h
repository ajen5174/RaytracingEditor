#pragma once
#include "Triangle.h"
#include "../Renderer/Hittable.h"
#include "../Core/Entity.h"


inline __global__ void CreateBVHNode(Hittable* node, Triangle** list, int numTriangles);
inline bool CompareBox(Triangle* left, Triangle* right);


class BVHNode : public Hittable
{
public:
	__device__ BVHNode() : Hittable() {}

	//__device__ BVHNode(Triangle** list, int numEntities) //right now this will only make bvh's for meshes
	//	:BVHNode(list, 0, numEntities) {}

	__device__ BVHNode(Triangle** newList, int start, int end)
		: Hittable()
	{
		//copy array into new array? not sure if I need to
		//Triangle** newList = list;
		//randomly select an axis?
		//int axis = 0;

		int objectSpan = end - start;

		if (objectSpan == 1)
		{
			left = right = (newList[start]);
		}
		else if (objectSpan == 2)
		{
			if (CompareBox(newList[start], newList[start + 1]))
			{
				left = newList[start];
				right = newList[start + 1];
			}
			else
			{
				left = newList[start + 1];
				right = newList[start];
			}
		}
		else
		{
			//sort of some kind
			//auto comparator = CompareBox;


			int midPoint = start + objectSpan / 2;

			left = new BVHNode(newList, start, midPoint);
			right = new BVHNode(newList, midPoint, end);
		}

		AABB leftBox, rightBox;

		left->BoundingBox(leftBox);
		right->BoundingBox(rightBox);

		box = SurroundingBox(leftBox, rightBox);
	}

	//void CreateTree(std::vector<Hittable*>& list, int start, int end)
	//{
	//	//copy array into new array? not sure if I need to
	//	std::vector<Hittable*> newList = list;
	//	//randomly select an axis?
	//	int axis = 0;

	//	int objectSpan = end - start;

	//	if (objectSpan == 1)
	//	{
	//		left = right = (newList[start]);
	//	}
	//	else if (objectSpan == 2)
	//	{
	//		if (CompareBox(newList[start], newList[start + 1]))
	//		{
	//			left = newList[start];
	//			right = newList[start + 1];
	//		}
	//		else
	//		{
	//			left = newList[start + 1];
	//			right = newList[start];
	//		}
	//	}
	//	else
	//	{
	//		auto comparator = CompareBox;

	//		//std::sort(newList.begin() + start, newList.begin() + end, comparator);

	//		int midPoint = start + objectSpan / 2;

	//		//CheckCudaErrors(cudaMallocManaged(&(left), sizeof(BVHNode)));
	//		//CheckCudaErrors(cudaMallocManaged(&(right), sizeof(BVHNode)));
	//		//CheckCudaErrors(cudaDeviceSynchronize());


	//		//CreateBVHNode<<<1, 1>>>(left);
	//		//CheckCudaErrors(cudaDeviceSynchronize());
	//		//CreateBVHNode<<<1, 1>>>(right);
	//		//CheckCudaErrors(cudaDeviceSynchronize());
	//		//((BVHNode*)(left))->CreateTree(newList, start, midPoint);
	//		//((BVHNode*)(right))->CreateTree(list, midPoint, end);
	//		left = new BVHNode(list, start, midPoint);
	//		right = new BVHNode(list, midPoint, end);
	//	}

	//	AABB leftBox, rightBox;

	//	left->BoundingBox(leftBox);
	//	right->BoundingBox(rightBox);

	//	box = SurroundingBox(leftBox, rightBox);
	//}


public:
	virtual __host__ __device__ bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) override
	{
		if(!box.Hit(ray, minDist, maxDist))
			return false;

		bool hitLeft = left->Hit(ray, minDist, maxDist, hitInfo); //recursive nonsense
		float newMaxDist = hitLeft ? hitInfo.distance : maxDist;
		bool hitRight = right->Hit(ray, minDist, newMaxDist, hitInfo);

		return hitLeft || hitRight;
	}

	virtual __device__ bool BoundingBox(AABB& outputBox) override
	{
		outputBox.min.x = box.min.x;
		outputBox.min.y = box.min.y;
		outputBox.min.z = box.min.z;

		outputBox.max.x = box.max.x;
		outputBox.max.y = box.max.y;
		outputBox.max.z = box.max.z;
		return true;
	}


public:
	AABB box;
	Hittable* left; //nodes can be BVHnodes or any other hittable
	Hittable* right;
};

inline __global__ void CreateBVHNode(BVHNode** node, Triangle** list, int numTriangles)
{
	(*node) = new BVHNode(list, 0, numTriangles);
}


inline __device__ bool CompareBox(Triangle* left, Triangle* right)
{
	AABB box1, box2;

	left->BoundingBox(box1);
	right->BoundingBox(box2);

	//axis is 0, 1 or 2 and corresponds to x, y and z
	return (box1.min[0] < box2.min[0]);
}