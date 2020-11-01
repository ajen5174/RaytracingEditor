#pragma once
#include "../Math/mat4.h"
#include "Json.h"

class Entity;

class Transform
{

public:
	__device__ Transform() : translation(0.0f), rotation(vec3(0.0f)), scale(1.0f) {}
	__device__ Transform(const vec3& translation, const vec3& rotation = vec3(0.0f), const vec3& scale = vec3(1.0f))
		:translation(translation), rotation(rotation), scale(scale) {}

	inline mat4 __device__ GetMatrix() const { return *this; };

	__device__ inline operator mat4() const
	{
		return mat4(translation, rotation, scale);
	};

	//inline Transform& operator=(const glm::mat4& matrix)
	//{
	//	translation = vec3(matrix[3]);
	//	rotation = quat_cast(matrix);
	//	scale = vec3(matrix[0][0], matrix[1][1], matrix[2][2]);

	//	return *this;
	//};

	//inline Transform* Clone()
	//{
	//	return new Transform(owner, translation, rotation, scale);
	//}


	bool Load(rapidjson::Value& value)
	{
		json::GetVec3(value, "translation", translation);
		json::GetVec3(value, "rotation", rotation);
		json::GetVec3(value, "scale", scale);


		return true;
	}

public:
	vec3 translation;
	vec3 rotation;
	vec3 scale;

	//Entity* owner;


};
