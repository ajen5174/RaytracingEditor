#include "Transform.h"
#include "../Core/Json.h"

std::string Transform::ToString()
{
	std::string str = std::to_string(translation.x) + " " + std::to_string(translation.y) + " " + std::to_string(translation.z);

	return str;
}

void Transform::Destroy()
{
}

bool Transform::Load(const rapidjson::Value& value)
{
	glm::vec3 translation;
	json::GetVec3(value, "translation", translation);

	this->translation = translation;

	glm::vec3 rotation;
	json::GetVec3(value, "rotation", rotation);
	this->rotation = glm::radians(rotation);

	glm::vec3 scale;
	json::GetVec3(value, "scale", scale);
	this->scale = scale;
	return true;
}

void Transform::Initialize()
{
}

void Transform::Update()
{
}

void Transform::BuildJSON(rapidjson::Value& v, rapidjson::MemoryPoolAllocator<>& mem)
{
	json::BuildCString(v, "type", "Transform", mem);
	json::BuildVec3(v, "translation", translation, mem);
	json::BuildVec3(v, "rotation", glm::degrees(glm::eulerAngles(rotation)), mem);
	json::BuildVec3(v, "scale", scale, mem);
}
