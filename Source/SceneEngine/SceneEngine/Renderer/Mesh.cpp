#include "Mesh.h"

Mesh::Mesh(std::string filename)
    : Object(filename.c_str())
{
    
}

Mesh::~Mesh()
{
}

void Mesh::Draw()
{
    vertexArray.Draw();
}

void Mesh::Destroy()
{
}

bool Mesh::Load(const rapidjson::Value&)
{
    return false;
}

void Mesh::Initialize()
{
}
