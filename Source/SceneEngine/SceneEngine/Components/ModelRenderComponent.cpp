#include "ModelRenderComponent.h"
#include "../Core/Scene.h"


void ModelRenderComponent::Update()
{
}

void ModelRenderComponent::Draw()
{
	Scene* scene = owner->GetScene();

	Camera* cam = scene->GetMainCamera();

	glm::mat4 modelViewMatrix = cam->viewMatrix * owner->GetComponent<Transform>()->GetMatrix();
	glm::mat4 mvpMatrix = cam->projectionMatrix * modelViewMatrix;

	//use shader

	model->Draw();
}
