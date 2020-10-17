#pragma once
#include <vector>
#include <glad/glad.h>
#include <glm/glm/glm.hpp>
#include <map>
#include "../Core/Object.h"

class Shader : public Object
{
public:
	Shader(StringId& name);
	~Shader();
	// Inherited via Object
	virtual void Destroy() override;
	virtual bool Load(const rapidjson::Value&) override;
	virtual void Initialize() override;

	void CreateFromFile(const std::string& filename, GLenum shaderType);
	void CreateFromSource(const std::string& source, GLenum shaderType);

	void Link();
	void Use();

	GLuint GetProgramId() { return shader; }
	bool IsLinked() { return linked; }

	void SetUniform(const std::string& name, float x, float y, float z);
	void SetUniform(const std::string& name, const glm::vec2& v2);
	void SetUniform(const std::string& name, const glm::vec3& v3);
	void SetUniform(const std::string& name, const glm::vec4& v4);
	void SetUniform(const std::string& name, const glm::mat4& mx4);
	void SetUniform(const std::string& name, const glm::mat3& mx3);
	void SetUniform(const std::string& name, float value);
	void SetUniform(const std::string& name, int value);
	void SetUniform(const std::string& name, bool value);
	void SetUniform(const std::string& name, GLuint value);

private:
	GLint GetUniform(const std::string& name);

private:
	GLuint shader;
	bool linked = false;
	std::map<std::string, GLint> uniforms;

	
};