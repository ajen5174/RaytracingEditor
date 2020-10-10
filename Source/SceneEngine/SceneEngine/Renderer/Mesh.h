#pragma once
#include <string>
#include "VertexArray.h"
#include "..\\Core\Object.h"

class Mesh : Object
{

public:
	Mesh(std::string filename);
	~Mesh();
	void Draw();

	// Inherited via Object
	virtual void Destroy() override;
	virtual bool Load(const rapidjson::Value&) override;
	virtual void Initialize() override;



private:
	VertexArray vertexArray;
	std::string directory;

	
};