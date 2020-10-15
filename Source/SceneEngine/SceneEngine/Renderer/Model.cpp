#include "Model.h"
#include "../Core/StringId.h"

Model::Model(std::string filename)
    :Object(filename.c_str())
{

}

Model::~Model()
{
}

void Model::Draw()
{
    mesh->Draw();
}

void Model::Destroy()
{
}

bool Model::Load(const rapidjson::Value&)
{
    return false;
}

void Model::Initialize()
{
}
