#include "Light.h"

void Light::Destroy()
{
}

bool Light::Load(const rapidjson::Value& v)
{
	json::GetFloat(v, "intensity", intensity);
	json::GetVec3(v, "color", color);
	json::GetString(v, "light_type", lightType);
	json::GetVec3(v, "direction", direction);
	return true;
}

void Light::BuildJSON(rapidjson::Value& v, rapidjson::MemoryPoolAllocator<>& mem)
{
	json::BuildCString(v, "type", "Light", mem);
	json::BuildVec3(v, "color", color, mem);
	json::BuildFloat(v, "intensity", intensity, mem);
	json::BuildString(v, "light_type", lightType, mem);
	json::BuildVec3(v, "direction", direction, mem);
}

void Light::Initialize()
{
}

void Light::Update()
{
}

void Light::SetShader(std::string lightname, Shader* shader)
{
	shader->Use();
	shader->SetUniform(lightname + ".position", owner->GetComponent<Transform>()->translation);
	shader->SetUniform(lightname + ".color", color);
	shader->SetUniform(lightname + ".intensity", intensity);
}
