#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include "Component.h"

#ifndef TRANSFORM_H
#define TRANSFORM_H
class Transform : public Component
{

public:
	Transform(StringId& name, Entity* owner) : translation(0.0f), rotation(glm::vec3(0.0f, 0.0f, 0.0f)), scale(1.0f), Component(name, owner){}
	Transform(StringId& name, Entity* owner, const glm::vec3& translation, const glm::quat& rotation = glm::quat(glm::vec3(0.0f)), const glm::vec3& scale = glm::vec3(1.0f))
		:translation(translation), rotation(rotation), scale(scale), Component(name, owner){}

	inline glm::mat4 GetMatrix() const { return *this; };

	inline operator glm::mat4() const 
	{
		glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scale);
		glm::mat4 rotationMatrix = glm::mat4_cast(rotation);
		glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), translation);

		return translationMatrix * rotationMatrix * scaleMatrix;
	};

	inline Transform& operator=(const glm::mat4& matrix)
	{
		translation = glm::vec3(matrix[3]);
		rotation = glm::quat_cast(matrix);
		scale = glm::vec3(matrix[0][0], matrix[1][1], matrix[2][2]);

		return *this;
	};

	inline Transform* Clone()
	{
		return new Transform(name, owner, translation, rotation, scale);
	}

	std::string ToString();

	// Inherited via Component
	virtual void Destroy() override;

	virtual bool Load(const rapidjson::Value&) override;

	virtual void Initialize() override;

	virtual void Update() override;

public:
	glm::vec3 translation;
	glm::quat rotation;
	glm::vec3 scale;


	

};
#endif