#pragma once
#include <glad/glad.h>
#include <vector>

class VertexArray
{
public:
	enum eAttrib
	{
		POSITION,
		NORMAL,
		COLOR,
		TEXCOORD,
		MULTI,
		TANGENT
	};

	struct VertexBuffer
	{
		eAttrib attrib;
		GLuint vbo;
		GLsizei num;
	};

public:
	VertexArray();
	~VertexArray();
	void CreateBuffer(eAttrib attrib, GLsizei size, GLsizei numVertex, void* data);
	void SetAttribute(eAttrib attrib, GLint numVertex, GLsizei stride, uint64_t offset);
	virtual void Draw(GLenum primitiveType = GL_TRIANGLES);
	void Bind() { glBindVertexArray(vao); };


protected:
	GLuint vao = 0;
	GLuint vertexCount = 0;
	std::vector<VertexBuffer> vertexBuffers;
};