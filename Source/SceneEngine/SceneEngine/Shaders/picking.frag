#version 330 core
out vec3 FragColor;

void main()
{
    FragColor = vec3(float(gl_PrimitiveID + 1), 0.0, 0.0);
} 
