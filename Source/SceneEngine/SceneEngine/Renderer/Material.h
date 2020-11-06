#pragma once
#include "../Core/Object.h"
#include "glm/glm/glm.hpp"
#include "../Core/Json.h"
#include "Shader.h"

class Material : public Object
{
public:

	Material(StringId& name) 
		: materialType("lambert"), albedo(0.8f), fuzz(0.0f), refractionIndex(0.0f), Object(name) {}

	void Use(Shader* shader);

	// Inherited via Object
	virtual void Destroy() override;
	virtual bool Load(const rapidjson::Value&) override;
	virtual void BuildJSON(rapidjson::Value& v, rapidjson::MemoryPoolAllocator<>& mem) override;
	virtual void Initialize() override;

public:
	std::string materialType; //lambert, metal, or dielectric for now
	glm::vec3 albedo; //lambert and metal
	float fuzz; //metal
	float refractionIndex; //dielectric
};