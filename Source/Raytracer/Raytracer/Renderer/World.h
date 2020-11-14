//#pragma once
//#include "../Core/Entity.h"
//#include "Light.h"
//
//
//class World : public Hittable
//{
//public:
//	// Inherited via Hittable
//	__host__ __device__ bool World::Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo)
//	{
//        float closestSoFar = maxDist;
//        bool hit = false;
//        HitInfo tempInfo;
//        for (int j = 0; j < numEntities; j++)
//        {
//            if (entities[j]->mesh)
//            {
//                //if (list[j]->mesh->boundingSphere->Hit(r, 0.0f, 100.0f, info)) //bug here for suzanne
//                {
//                    for (int i = 0; i < entities[j]->mesh->numTriangles; i++)
//                    {
//                        if (entities[j]->mesh->triangles[i]->Hit(ray, 0.001f, closestSoFar, tempInfo))
//                        {
//                            hitInfo = tempInfo;
//                            closestSoFar = hitInfo.distance;
//                            hit = true;
//
//                        }
//                    }
//                }
//            }
//        }
//        return hit;
//	}
//
//	__host__ __device__ bool World::BoundingBox(float time0, float time1, AABB& outputBox)
//	{
//        if (numEntities < 1) return false;
//
//        AABB tempBox;
//        bool firstBox = true;
//
//        for (int i = 0; i < numEntities; i++)
//        {
//            
//        }
//
//		return true;
//	}
//
//
//
//
//public:
//	Entity** entities;
//	Light** lights;
//	int numEntities;
//	int numLights;
//
//
//
//	
//};