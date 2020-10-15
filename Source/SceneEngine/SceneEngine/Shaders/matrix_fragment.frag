#version 330 core
out vec4 FragColor;

in vec3 ourColor;

uniform mat4 myModel;
uniform mat4 view;
uniform mat4 projection;

void main()
{

	FragColor = vec4(ourColor, 1.0);
}