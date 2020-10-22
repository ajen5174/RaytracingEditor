#pragma once
#include "../Math/Ray.h"

struct HitInfo
{
	float distance;
	vec3 point;
	vec3 normal;
	//Material* material;
};

class Hittable
{
public:
	virtual bool Hit(Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) = 0;
};