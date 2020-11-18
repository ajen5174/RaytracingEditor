#pragma once
#include <rapidjson/document.h>
#include "Transform.h"
#include "Json.h"
#include "../Renderer/Mesh.h"
#include "../Math/Sphere.h"
#include "../Renderer/Light.h"

inline __global__ void CreateTransform(Transform* transform)
{
	transform = new Transform();
}

inline __global__ void CreateLight(Light* light)
{
	light = new Light();
}

inline __global__ void CreateSphere(Sphere* sphere)
{
	sphere = new Sphere(vec3(0.0f), 1.0f);
}

inline __global__ void CreateCamera(Camera* cam)
{
	cam = new Camera();
}

inline __global__ void CreateMesh(Mesh* mesh)
{
	mesh = new Mesh();
	
}

inline __global__ void CreateMaterial(Material* material)
{
	material = new Material();
}

//This method uses data read in from disk and stored in vertices to create our device only Triangles for collision. W
	//We need this method because the Triangle MUST be "new'd" on the DEVICE, or the vtable (hittable) won't be accessible by the device later.
//inline __global__ void CreateMeshTriangles(Mesh* mesh, Transform* transform)
//{
//	//transform the vertices lmao
//	mat4 t = transform->GetMatrix();
//
//	for (int i = 0; i < mesh->numTriangles * 3; i++)
//	{
//		mesh->vertices[i] = t * mesh->vertices[i];
//	}
//
//	for (int i = 0; i < mesh->numTriangles; i++)
//	{
//		//if (i % 3 == 0)
//		{
//			mesh->triangles[i] = new Triangle(mesh->vertices[(i * 3) + 0], mesh->vertices[(i * 3) + 1], mesh->vertices[(i * 3) + 2]);
//			mesh->triangles[i]->material = mesh->material;
//		}
//	}
//
//	vec3 averagePosition;
//	averagePosition = vec3();
//	for (int i = 0; i < mesh->numTriangles * 3; i++)
//	{
//		averagePosition = averagePosition + mesh->vertices[i];
//	}
//	averagePosition = averagePosition / (mesh->numTriangles * 3);
//	float boundingRadius = 0.0f;
//	for (int i = 0; i < mesh->numTriangles * 3; i++)
//	{
//		float tempRadius = (averagePosition - mesh->vertices[i]).Magnitude();
//		if (tempRadius > boundingRadius)
//			boundingRadius = tempRadius;
//	}
//	mesh->boundingSphere = new Sphere(averagePosition, boundingRadius);
//
//}

enum ModelType
{
	POLYGON_MESH,
	SPHERE
};

class Entity
{
public:
	__device__ Entity()
	{
		mesh = new Mesh(); 
		transform = new Transform();
	}

	void Destroy()
	{

	}

	__host__ bool Load(rapidjson::Value& value)
	{
		//ensure transform is complete first
		rapidjson::Value& components = value["components"];

		for (int i = 0; i < components.Size(); i++)
		{
			rapidjson::Value& componentValue = components[i];
			std::string componentType;

			json::GetString(componentValue, "type", componentType);

			if (componentType == "Transform")
			{
				//StringId temp = name.ToString() + "Transform";
				//Transform* transform = new Transform(this);
				Transform* transform;
				CheckCudaErrors(cudaMallocManaged(&transform, sizeof(Transform)));
				CreateTransform << <1, 1 >> > (transform);
				CheckCudaErrors(cudaDeviceSynchronize());
				if (transform->Load(componentValue))
				{
					this->transform = transform;
				}
				break;
			}
		}

		if (components.IsArray())
		{
			for (int i = 0; i < components.Size(); i++)
			{
				rapidjson::Value& componentValue = components[i];
				std::string componentType;

				json::GetString(componentValue, "type", componentType);

				//if (componentType == "Transform")
				//{
				//	//StringId temp = name.ToString() + "Transform";
				//	//Transform* transform = new Transform(this);
				//	Transform* transform;
				//	CheckCudaErrors(cudaMallocManaged(&transform, sizeof(Transform)));
				//	CreateTransform<<<1, 1>>> (transform);
				//	CheckCudaErrors(cudaDeviceSynchronize());
				//	if (transform->Load(componentValue))
				//	{
				//		this->transform = transform;
				//	}
				//}
				//else 
				if (componentType == "ModelRender")
				{
					std::string materialType;
					json::GetString(componentValue, "materialType", materialType);

					Material* material;
					CheckCudaErrors(cudaMallocManaged(&material, sizeof(Material)));
					CreateMaterial << <1, 1 >> > (material);
					CheckCudaErrors(cudaDeviceSynchronize());

					if (materialType == "lambert")
					{
						json::GetVec3(componentValue, "albedo", material->albedo);
						material->materialType = 'l';
					}
					else if (materialType == "metal")
					{
						json::GetVec3(componentValue, "albedo", material->albedo);
						json::GetFloat(componentValue, "fuzz", material->fuzz);
						material->materialType = 'm';
					}
					else if (materialType == "dielectric")
					{
						json::GetFloat(componentValue, "refractionIndex", material->refractionIndex);
						material->materialType = 'd';
					}

					int modelType = (int)ModelType::POLYGON_MESH;
					json::GetInt(componentValue, "modelType", modelType);
					if (modelType == (int)ModelType::POLYGON_MESH)
					{
						Mesh* mesh;
						CheckCudaErrors(cudaMallocManaged(&mesh, sizeof(Mesh)));
						CreateMesh << <1, 1 >> > (mesh);
						CheckCudaErrors(cudaDeviceSynchronize());
						mesh->parentTransform = this->transform;
						mesh->material = material; 
						if (mesh->Load(componentValue))
						{
							this->mesh = mesh;
						}
					}
					else if(modelType == (int)ModelType::SPHERE)
					{
						Sphere* sphere;
						CheckCudaErrors(cudaMallocManaged(&sphere, sizeof(Sphere)));
						CreateSphere<<<1, 1>>>(sphere);
						CheckCudaErrors(cudaDeviceSynchronize());
						sphere->center = transform->translation;
						sphere->radius = fmax(transform->scale.x, transform->scale.y);
						sphere->radius = fmax(sphere->radius, transform->scale.z);
						sphere->material = material;
						this->primitive = sphere;
					}
					
					
				}
				else if (componentType == "Camera")
				{
					Camera* cam;
					CheckCudaErrors(cudaMallocManaged(&cam, sizeof(Camera)));
					CreateCamera << <1, 1 >> > (cam);
					CheckCudaErrors(cudaDeviceSynchronize());
					if (cam->Load(componentValue))
					{
						this->cam = cam;
						this->cam->SetView(this->transform->translation, this->transform->translation + vec3(0.0f, 0.0f, -1.0f), vec3(0.0f, 1.0f, 0.0f)); //matches opengl
					}
				}
				else if (componentType == "Light")
				{
					Light* light;
					CheckCudaErrors(cudaMallocManaged(&light, sizeof(Light)));
					CreateLight<<<1, 1>>>(light);
					CheckCudaErrors(cudaDeviceSynchronize());
					if (light->Load(componentValue))
					{
						this->light = light;
						this->light->owner = this;
					}
				}
			}

			//if (mesh)
			//{
			//	CreateMeshTriangles << <1, 1 >> > (mesh, transform);
			//	CheckCudaErrors(cudaDeviceSynchronize());
			//}
			
			return true;
		}
	}

	

	//virtual void Initialize() = 0;

public:
	Transform* transform;
	Camera* cam;
	Mesh* mesh;
	Sphere* primitive;
	Light* light;

};