#include "VertexIndexArray.h"

VertexIndexArray::VertexIndexArray()
{
}

VertexIndexArray::~VertexIndexArray()
{
	glDeleteBuffers(1, &ibo);
}

void VertexIndexArray::CreateIndexBuffer(GLenum indexType, GLsizei numIndex, void* data)
{
	this->indexType = indexType;
	this->indexCount = numIndex;
	glGenBuffers(1, &ibo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
	uint32_t indexSize = (indexType == GL_UNSIGNED_SHORT) ? sizeof(GLushort) : sizeof(GLuint);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexSize * numIndex, data, GL_STATIC_DRAW);
}

void VertexIndexArray::Draw(GLenum primitiveType)
{
	glBindVertexArray(vao);
	glDrawElements(primitiveType, indexCount, indexType, 0);
	glBindVertexArray(0);
}
