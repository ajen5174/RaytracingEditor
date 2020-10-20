#include "RenderComponent.h"
#include "../Renderer/Model.h"
#include "../Renderer/Shader.h"

#ifndef MODEL_RENDER_COMPONENT_H
#define MODEL_RENDER_COMPONENT_H
class ModelRenderComponent : public RenderComponent
{
public:
	ModelRenderComponent(StringId& name, Entity* owner);
	// Inherited via RenderComponent
	virtual void Update() override;
	virtual void Draw() override;

private:
	Model* model;
	Shader* shader;
};



#endif