#version 330 core
out vec3 FragColor;

uniform float objectID;

void main()
{
    FragColor = vec3(objectID, 0, (gl_PrimitiveID + 1));
} 
