#version 330 core
out vec4 FragColor;

in vec3 fposition;
in vec3 fnormal;

uniform vec3 lightPosition;


void main()
{
	vec3 positionToLight = normalize(vec3(lightPosition) - fposition);
	
	vec3 ambient = vec3(0.3);

	float lDotN = max(0.0, dot(positionToLight, fnormal));
	vec3 diffuse = ambient * lDotN; //this is super wrong but works for now?

	FragColor = vec4(ambient + diffuse, 1.0);
	
	//FragColor = vec4(ourColor, 1.0);
}