#pragma once
#include "../Math/vec3.h"

class Light 
{
public:
	enum LightType
	{
		POINT,
		DIRECTION,
		SPOTLIGHT
	};

	__device__ Light() : color(0.0f), intensity(0.0f) 
	{
		
	}
	__device__ Light(vec3 inColor, float inIntensity) : color(inColor), intensity(inIntensity) {}


	bool Load(rapidjson::Value& value)
	{
		json::GetVec3(value, "color", color);
		json::GetFloat(value, "intensity", intensity);
		std::string type;
		json::GetString(value, "light_type", type);
		if (type == "point")
			lightType = LightType::POINT;
		else if (type == "direction")
			lightType = LightType::DIRECTION;
		else if (type == "spotlight")
			lightType = LightType::SPOTLIGHT;
		json::GetVec3(value, "direction", direction);
		return true;
	}



public:
	vec3 color;
	float intensity;
	LightType lightType;
	vec3 direction;

	Entity* owner;
};