#pragma once
#include <string>
#include "..\\Core\Object.h"
#include "Mesh.h"
#include "Material.h"

class Model : Object
{
public:
	Model(StringId name);
	~Model();
	void Draw();

	// Inherited via Object
	virtual void Destroy() override;
	virtual bool Load(const rapidjson::Value&) override;
	bool ReloadMesh(std::string path);
	virtual void Initialize() override;
	virtual void BuildJSON(rapidjson::Value& v, rapidjson::MemoryPoolAllocator<>& mem) override;


public:
	//texture
	Material* material;
	Mesh* mesh;
	
};