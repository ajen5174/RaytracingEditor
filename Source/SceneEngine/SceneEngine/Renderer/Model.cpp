#include "Model.h"
#include "../Core/StringId.h"
#include "../EngineLibrary.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

Model::Model(std::string name)
    :Object(name.c_str())
{
    rapidjson::Value v;
    Load(v);
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

bool Model::Load(const rapidjson::Value& file)
{

#ifdef _WINDLL
    std::string modelPath("..\\..\\..\\..\\..\\Content\\Meshes\\suzanne.obj");
#else
    std::string modelPath("C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\Content\\Meshes\\suzanne.obj");
#endif

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(modelPath, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        PrintDebugMessage("Error loading model: " + std::string(importer.GetErrorString()));
        return false;
    }

    PrintDebugMessage("File loaded");
    if (scene->mNumMeshes > 1)
    {
        PrintDebugMessage("Too many meshes in this file, the first mesh will be loaded.");
    }

    auto loadedMesh = scene->mMeshes[0];
    
    mesh = new Mesh(modelPath);
    mesh->Load(loadedMesh);


    return false;
}

void Model::Initialize()
{
}
