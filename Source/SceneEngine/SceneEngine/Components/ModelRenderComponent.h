#include "RenderComponent.h"
#include "../Renderer/Model.h"

#ifndef MODEL_RENDER_COMPONENT_H
#define MODEL_RENDER_COMPONENT_H
class ModelRenderComponent : public RenderComponent
{
public:
	ModelRenderComponent(StringId& name, Entity* owner) : RenderComponent(name, owner) {}
	// Inherited via RenderComponent
	virtual void Update() override;
	virtual void Draw() override;

private:
	Model* model;
};



#endif