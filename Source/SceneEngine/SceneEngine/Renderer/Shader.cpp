#include "Shader.h"
#include <iostream>
#include <glm/glm/gtc/type_ptr.hpp>
#include "../EngineLibrary.h"

Shader::~Shader()
{
	if (shader == 0)
		return;

	GLint numShaders = 0;
	glGetProgramiv(shader, GL_ATTACHED_SHADERS, &numShaders);
	std::vector<GLuint> shaders(numShaders);
	glGetAttachedShaders(shader, numShaders, NULL, shaders.data());

	for (GLuint s : shaders)
	{
		glDetachShader(shader, s);
		glDeleteShader(s);
	}

	glDeleteProgram(shader);
}

void Shader::CreateFromFile(const std::string& filename, GLenum shaderType)
{
	//read file in to a string
	std::string source;

	CreateFromSource(source, shaderType);
}

void Shader::CreateFromSource(const std::string& source, GLenum shaderType)
{
	GLuint s = glCreateShader(shaderType);

	const char* sourceC = source.c_str();
	glShaderSource(s, 1, &sourceC, NULL);
	glCompileShader(s);

	int success;

	glGetShaderiv(s, GL_COMPILE_STATUS, &success);
	if (!success) {
		int length = 0;
		glGetShaderiv(s, GL_INFO_LOG_LENGTH, &length);
		if (length > 0)
		{
			std::string infoLog(length, ' ');
			glGetShaderInfoLog(s, length, &length, &infoLog[0]);
			std::cout << "ERROR::VERTEX::SHADER::COMPILE_FAILED\n" << infoLog << std::endl;
			PrintDebugMessage(infoLog);
		}

		glDeleteShader(s);
		
	}
	else
	{
		glAttachShader(shader, s);
	}
}

void Shader::Link()
{
	if (linked)
		return;

	glLinkProgram(shader);

	GLint success;
	glGetProgramiv(shader, GL_LINK_STATUS, &success);
	if (!success) {

		int length = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
		if (length > 0)
		{
			std::string infoLog(length, ' ');
			glGetProgramInfoLog(shader, length, &length, &infoLog[0]);
			std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
			PrintDebugMessage(std::string(infoLog));
		}
		glDeleteProgram(shader);
		
	}
	else
	{
		linked = true;
	}
}

void Shader::Use()
{
	if (shader && linked)
		glUseProgram(shader);
}

void Shader::SetUniform(const std::string& name, float x, float y, float z)
{
	GLint uniform = GetUniform(name);
	glUniform3f(uniform, x, y, z);
}

void Shader::SetUniform(const std::string& name, const glm::vec2& v2)
{
	GLint uniform = GetUniform(name);
	glUniform2f(uniform, v2.x, v2.y);
}

void Shader::SetUniform(const std::string& name, const glm::vec3& v3)
{
	GLint uniform = GetUniform(name);
	glUniform3f(uniform, v3.x, v3.y, v3.z);
}

void Shader::SetUniform(const std::string& name, const glm::vec4& v4)
{
	GLint uniform = GetUniform(name);
	glUniform4f(uniform, v4.x, v4.y, v4.z, v4.w);
}

void Shader::SetUniform(const std::string& name, const glm::mat4& mx4)
{
	GLint uniform = GetUniform(name);
	glUniformMatrix4fv(uniform, 1, GL_FALSE, glm::value_ptr(mx4));
}

void Shader::SetUniform(const std::string& name, const glm::mat3& mx3)
{
	GLint uniform = GetUniform(name);
	glUniformMatrix3fv(uniform, 1, GL_FALSE, glm::value_ptr(mx3));
}

void Shader::SetUniform(const std::string& name, float value)
{
	GLint uniform = GetUniform(name);
	glUniform1f(uniform, value);
}

void Shader::SetUniform(const std::string& name, int value)
{
	GLint uniform = GetUniform(name);
	glUniform1i(uniform, value);
}

void Shader::SetUniform(const std::string& name, bool value)
{
	GLint uniform = GetUniform(name);
	glUniform1i(uniform, value);
}

void Shader::SetUniform(const std::string& name, GLuint value)
{
	GLint uniform = GetUniform(name);
	glUniform1ui(uniform, value);
}

GLint Shader::GetUniform(const std::string& name)
{
	auto uniform = uniforms.find(name);
	if (uniform == uniforms.end())
	{
		uniforms[name] = glGetUniformLocation(shader, name.c_str());
	}

	return uniforms[name];
}
