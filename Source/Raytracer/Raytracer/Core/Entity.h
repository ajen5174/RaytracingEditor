#pragma once
#include <rapidjson/document.h>
#include "Transform.h"
#include "Json.h"
#include "../Renderer/Mesh.h"
#include "../Math/Sphere.h"

inline __global__ void CreateTransform(Transform* transform)
{
	transform = new Transform();
}


inline __global__ void CreateMesh(Mesh* mesh)
{
	mesh = new Mesh();
}

//This method uses data read in from disk and stored in vertices to create our device only Triangles for collision. W
	//We need this method because the Triangle MUST be "new'd" on the DEVICE, or the vtable (hittable) won't be accessible by the device later.
inline __global__ void CreateMeshTriangles(Mesh* mesh, Transform* transform)
{
	//transform the vertices lmao
	mat4 t = transform->GetMatrix();

	for (int i = 0; i < mesh->numTriangles * 3; i++)
	{
		mesh->vertices[i] = t * mesh->vertices[i];
	}

	for (int i = 0; i < mesh->numTriangles; i++)
	{
		//if (i % 3 == 0)
		{
			mesh->triangles[i] = new Triangle(mesh->vertices[(i * 3) + 0], mesh->vertices[(i * 3) + 1], mesh->vertices[(i * 3) + 2]);
		}
	}

	vec3 averagePosition;
	averagePosition = vec3();
	for (int i = 0; i < mesh->numTriangles * 3; i++)
	{
		averagePosition = averagePosition + mesh->vertices[i];
	}
	averagePosition = averagePosition / (mesh->numTriangles * 3);
	float boundingRadius = 0.0f;//arbitrarily large number, should be float max, but Im not sure what the macro for that is.
	for (int i = 0; i < mesh->numTriangles * 3; i++)
	{
		float tempRadius = (averagePosition - mesh->vertices[i]).Magnitude();
		if (tempRadius > boundingRadius)
			boundingRadius = tempRadius;
	}
	mesh->boundingSphere = new Sphere(averagePosition, boundingRadius);

}

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
		rapidjson::Value& components = value["components"];
		if (components.IsArray())
		{
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
					CreateTransform<<<1, 1>>> (transform);
					CheckCudaErrors(cudaDeviceSynchronize());
					if (transform->Load(componentValue))
					{
						this->transform = transform;
					}
				}
				else if (componentType == "ModelRender")
				{
					Mesh* mesh;
					CheckCudaErrors(cudaMallocManaged(&mesh, sizeof(Transform)));
					CreateMesh << <1, 1 >> > (mesh);
					CheckCudaErrors(cudaDeviceSynchronize());
					if (mesh->Load(componentValue))
					{
						
						this->mesh = mesh;
					}
				}
				else if (componentType == "Camera")
				{
					
				}
			}

			if (mesh)
			{
				CreateMeshTriangles << <1, 1 >> > (mesh, transform);
				CheckCudaErrors(cudaDeviceSynchronize());
			}
			
			return true;
		}
	}
	//virtual void Initialize() = 0;

public:
	Transform* transform;
	//Camera cam;
	Mesh* mesh;

};