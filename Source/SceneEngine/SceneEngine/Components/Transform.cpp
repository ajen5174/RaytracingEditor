#include "Transform.h"

std::string Transform::ToString()
{
	std::string str = std::to_string(translation.x) + " " + std::to_string(translation.y) + " " + std::to_string(translation.z);

	return str;
}

void Transform::Destroy()
{
}

bool Transform::Load(const rapidjson::Value&)
{
	return false;
}

void Transform::Initialize()
{
}

void Transform::Update()
{
}
