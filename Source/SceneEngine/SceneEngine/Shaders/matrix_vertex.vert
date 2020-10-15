#version 330 core
layout (location = 0) in vec3 aPos;   // the position variable has attribute position 0

//struct material_s
//{
//	vec4 position;
//	vec3 ambient;
//	vec3 diffuse;
//	vec3 specular;
//};
//
//uniform material_s material;
uniform mat4 mvpMatrix;

 
out vec3 ourColor; // output a color to the fragment shader

void main()
{
	gl_Position = mvpMatrix * vec4(aPos, 1.0);
    ourColor = vec3(0.5, 0.2, 0.7);//material.diffuse; // set ourColor to the input color we got from the vertex data
}