#include "Material.h"

void Material::Use(Shader* shader)
{
	shader->Use();

	shader->SetUniform("material.diffuse", albedo);

	//texture nonsense?
}

void Material::Destroy()
{
}

bool Material::Load(const rapidjson::Value& v)
{
	json::GetString(v, "materialType", materialType);
	if (materialType == "lambert" || materialType == "metal")
	{
		json::GetVec3(v, "albedo", albedo);
	}
	if (materialType == "metal")
	{
		json::GetFloat(v, "fuzz", fuzz);
	}
	if (materialType == "dielectric")
	{
		json::GetFloat(v, "refractionIndex", refractionIndex);
	}

	return true;
}

void Material::BuildJSON(rapidjson::Value& v, rapidjson::MemoryPoolAllocator<>& mem)
{
	json::BuildString(v, "materialType", materialType, mem);
	json::BuildVec3(v, "albedo", albedo, mem);
	json::BuildFloat(v, "fuzz", fuzz, mem);
	json::BuildFloat(v, "refractionIndex", refractionIndex, mem);
	
}

void Material::Initialize()
{
}
