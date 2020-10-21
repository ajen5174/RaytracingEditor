#version 330 core
layout (location = 0) in vec3 vposition;
layout (location = 1) in vec3 vnormal;

uniform mat4 mvp;

void main()
{
	gl_Position = mvp * vec4(vposition + vnormal * 0.1, 1.0);
}
