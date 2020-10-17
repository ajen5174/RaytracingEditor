#include "VertexArray.h"
//#include <string>
#include "../EngineLibrary.h"

VertexArray::VertexArray()
{
	glGenVertexArrays(1, &vao);
	Bind();
}

VertexArray::~VertexArray()
{
	glDeleteVertexArrays(1, &vao);

	for (VertexBuffer vb : vertexBuffers)
	{
		glDeleteBuffers(1, &vb.vbo);
	}
}

void VertexArray::CreateBuffer(eAttrib attrib, GLsizei size, GLsizei numVertex, void* data)
{
	vertexCount = numVertex;
	GLuint vbo = 0;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
	VertexBuffer vertexBuffer = { attrib, vbo, numVertex };
	vertexBuffers.push_back(vertexBuffer);
}

void VertexArray::SetAttribute(eAttrib attrib, GLint numVertex, GLsizei stride, uint64_t offset)
{
	glEnableVertexAttribArray(attrib);
	glVertexAttribPointer(attrib, numVertex, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offset));

}

void VertexArray::Draw(GLenum primitiveType)
{
	glBindVertexArray(vao);
	glDrawArrays(primitiveType, 0, vertexCount);
	glBindVertexArray(0);
}
