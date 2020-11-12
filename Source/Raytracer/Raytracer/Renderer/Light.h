#pragma once
#include "../Math/vec3.h"

class Light 
{
public:
	__device__ Light() : color(0.0f), intensity(0.0f) 
	{
		
	}
	__device__ Light(vec3 inColor, float inIntensity) : color(inColor), intensity(inIntensity) {}


	bool Load(rapidjson::Value& value)
	{
		json::GetVec3(value, "color", color);
		json::GetFloat(value, "intensity", intensity);
		return true;
	}



public:
	vec3 color;
	float intensity;

	Entity* owner;
};