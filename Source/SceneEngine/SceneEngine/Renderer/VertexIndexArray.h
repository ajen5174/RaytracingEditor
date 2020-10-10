#pragma once
#include "VertexArray.h"

class VertexIndexArray : VertexArray
{
public:
	VertexIndexArray();
	~VertexIndexArray();
	void CreateIndexBuffer(GLenum indexType, GLsizei numIndex, void* data);

	virtual void Draw(GLenum primitiveType = GL_TRIANGLES) override;

protected:
	GLuint ibo = 0;
	GLuint indexCount = 0;
	GLenum indexType = 0;
};