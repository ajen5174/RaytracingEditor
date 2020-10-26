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
	rotation = glm::vec3(0.0f, 0.0f, glm::radians(90.0f));
	return true;
}

void Transform::Initialize()
{
}

void Transform::Update()
{
}
