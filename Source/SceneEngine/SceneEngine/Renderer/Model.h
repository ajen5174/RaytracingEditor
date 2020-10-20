#pragma once
#include <string>
#include "..\\Core\Object.h"
#include "Mesh.h"

class Model : Object
{
public:
	Model(std::string name);
	~Model();
	void Draw();

	// Inherited via Object
	virtual void Destroy() override;
	virtual bool Load(const rapidjson::Value&) override;
	virtual void Initialize() override;



public:
	//texture
	//material
	Mesh* mesh;

	
};