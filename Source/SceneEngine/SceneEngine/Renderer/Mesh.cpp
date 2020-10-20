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

bool Mesh::Load(const aiMesh* mesh)
{
    

    vertexArray.CreateBuffer(VertexArray::eAttrib::POSITION, (sizeof(float) * 3) * mesh->mNumVertices, mesh->mNumVertices, &mesh->mVertices[0].x);
    vertexArray.Bind();
    vertexArray.SetAttribute(VertexArray::eAttrib::POSITION, 3, 3 * sizeof(float), 0);



    return false;
}
