#include "../Components/Component.h"

#ifndef CAMERA_H
#define CAMERA_H
class Camera : public Component
{
public:
	void SetProjection(float fov, float aspectRatio, float nearClip, float farClip);
	// Inherited via Component
	virtual void Update() override;
public:
	glm::mat4 viewMatrix;
	glm::mat4 projectionMatrix;

private:
	float fov;
	float aspectRatio;
	float nearClip;
	float farClip;


};

#endif