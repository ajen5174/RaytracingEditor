#version 330 core
layout (location = 0) in vec3 vposition;   // the position variable has attribute position 0
layout (location = 1) in vec3 vnormal;

uniform mat4 mvp;
uniform mat4 mv;

 
out vec3 fposition; // output a color to the fragment shader
out vec3 fnormal;


void main()
{
    fnormal = normalize(mat3(mv) * vnormal);
    fposition = vec3(mv * vec4(vposition, 1.0));

	gl_Position = mvp *  vec4(vposition, 1.0);
}