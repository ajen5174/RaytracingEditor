#include "Model.h"
#include "../Core/StringId.h"
#include "../EngineLibrary.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "../Core/Json.h"

Model::Model(StringId name)
    :Object(name)
{
    material = new Material(name);
    mesh = new Mesh("");
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

bool Model::ReloadMesh(std::string path)
{
    if (modelType == ModelType::POLYGON_MODEL)
    {
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals);

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

        if (mesh) delete mesh;
        mesh = new Mesh(path);
        PrintDebugMessage("This mesh contains " + std::to_string(loadedMesh->mNumVertices) + " vertices.");
        return mesh->Load(loadedMesh);
    }
    else if (modelType == ModelType::SPHERE)
    {
        std::string spherePath = "../../../../../Content/Meshes/sphere.obj";

        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(spherePath, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            PrintDebugMessage("Error loading sphere model: " + std::string(importer.GetErrorString()));
            return false;
        }
        if (scene->mNumMeshes > 1)
        {
            PrintDebugMessage("Too many meshes in this file, the first mesh will be loaded.");
        }
        auto loadedMesh = scene->mMeshes[0];

        if (mesh) delete mesh;
        mesh = new Mesh(spherePath);

        return mesh->Load(loadedMesh);
    }
}

bool Model::Load(const rapidjson::Value& value)
{
    int tempType = -1;
    json::GetInt(value, "modelType", tempType);
    if (tempType > -1)
        modelType = (ModelType)tempType;

    if (modelType == ModelType::POLYGON_MODEL)
    {
        std::string modelPath;
        json::GetString(value, "meshPath", modelPath);

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

        if (mesh) delete mesh;
        mesh = new Mesh(modelPath);

        StringId matName = "material";
        if (material) delete material;
        material = new Material(matName);

        return mesh->Load(loadedMesh) && material->Load(value);
    }
    else if (modelType == ModelType::SPHERE)
    {
        std::string spherePath = "../../../../../Content/Meshes/sphere.obj";
        
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(spherePath, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            PrintDebugMessage("Error loading sphere model: " + std::string(importer.GetErrorString()));
            return false;
        }
        if (scene->mNumMeshes > 1)
        {
            PrintDebugMessage("Too many meshes in this file, the first mesh will be loaded.");
        }
        auto loadedMesh = scene->mMeshes[0];

        if (mesh) delete mesh;
        mesh = new Mesh(spherePath);

        StringId matName = "material";
        if (material) delete material;
        material = new Material(matName);

        return mesh->Load(loadedMesh) && material->Load(value);
    }
    

}

void Model::Initialize()
{
}

void Model::BuildJSON(rapidjson::Value& v, rapidjson::MemoryPoolAllocator<>& mem)
{
    json::BuildInt(v, "modelType", (int)modelType, mem);
    mesh->BuildJSON(v, mem);
    material->BuildJSON(v, mem);
}
