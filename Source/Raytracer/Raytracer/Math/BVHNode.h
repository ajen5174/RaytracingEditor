#pragma once
#include "Triangle.h"
#include "AABB.h"
#include "../Core/Entity.h"


inline __global__ void CreateBVHNode(Hittable* node, Triangle** list, int numTriangles);
inline bool CompareBox(Triangle* left, Triangle* right, int axis);
inline __device__ void QuickSortTriangles(Triangle** list, int start, int end, bool(*comparator)(Triangle*, Triangle*));
inline __device__ bool CompareBoxAxisX(Triangle* left, Triangle* right);
inline __device__ bool CompareBoxAxisY(Triangle* left, Triangle* right);
inline __device__ bool CompareBoxAxisZ(Triangle* left, Triangle* right);


class BVHNode : public Hittable
{
public:
	__device__ BVHNode() : Hittable() {}

	//__device__ BVHNode(Triangle** list, int numEntities) //right now this will only make bvh's for meshes
	//	:BVHNode(list, 0, numEntities) {}

	__device__ BVHNode(Triangle** newList, int start, int end, curandState& randState)
		: Hittable()
	{

		float axis = curand_uniform(&randState);
		axis *= 3;
		int intAxis = (int)axis;
		//sort of some kind
		auto comparator = axis == 0 ? CompareBoxAxisX : 
						  axis == 1 ? CompareBoxAxisY :
									  CompareBoxAxisZ;

		int objectSpan = end - start;

		if (objectSpan == 1)
		{
			left = right = (newList[start]);
		}
		else if (objectSpan == 2)
		{
			if (comparator(newList[start], newList[start + 1]))
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

			QuickSortTriangles(newList, start, end - 1, comparator);


			int midPoint = start + objectSpan / 2;

			left = new BVHNode(newList, start, midPoint, randState);
			right = new BVHNode(newList, midPoint, end, randState);
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

	virtual __host__ __device__ bool BoundingBox(AABB& outputBox) override
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
	curandState randState;
	curand_init(1984, 0, 0, &randState);
	(*node) = new BVHNode(list, 0, numTriangles, randState);
}


inline __device__ bool CompareBox(Triangle* left, Triangle* right, int axis)
{
	AABB box1, box2;

	left->BoundingBox(box1);
	right->BoundingBox(box2);

	//axis is 0, 1 or 2 and corresponds to x, y and z
	return (box1.min[axis] < box2.min[axis]);
}

inline __device__ bool CompareBoxAxisX(Triangle* left, Triangle* right)
{
	CompareBox(left, right, 0);
}

inline __device__ bool CompareBoxAxisY(Triangle* left, Triangle* right)
{
	CompareBox(left, right, 1);
}

inline __device__ bool CompareBoxAxisZ(Triangle* left, Triangle* right)
{
	CompareBox(left, right, 2);
}



inline __device__ void QuickSortTrianglesHelper(Triangle** list, int startIndex, int pivotIndex, bool(*comparator)(Triangle*, Triangle*))
{
	if (startIndex >= pivotIndex)
		return;
	else
	{
		int index = startIndex;
		for (int i = index; i < pivotIndex; i++)
		{
			if (comparator(list[i], list[pivotIndex]))
			{
				Triangle* temp = list[i];
				list[i] = list[index];
				list[index] = temp;
				index++;
			}
		}

		Triangle* temp2 = list[index];
		list[index] = list[pivotIndex];
		list[pivotIndex] = temp2;

		QuickSortTrianglesHelper(list, startIndex, index - 1, comparator);
		QuickSortTrianglesHelper(list, index + 1, pivotIndex, comparator);


	}
}

inline __device__ void QuickSortTriangles(Triangle** list, int start, int end, bool(*comparator)(Triangle*, Triangle*))
{
	QuickSortTrianglesHelper(list, start, end, comparator);
}