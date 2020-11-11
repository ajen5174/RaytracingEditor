#version 330 core
#define MAX_LIGHTS 15
out vec4 FragColor;

in vec3 fposition;
in vec3 fnormal;

//uniform vec3 lightPosition;

struct material_s
{
	vec3 diffuse;
};


struct light_s
{
	//int type;
	vec3 position;
	vec3 color;
	//vec3 direction;
	//vec3 ambient;
	//vec3 diffuse;
	//vec3 specular;
	float intensity;
	//float cutoff;
	//float exponent;
};

uniform light_s lights[MAX_LIGHTS];

uniform material_s material;

void main()
{
	vec3 finalColor;
	for(int i = 0; i < MAX_LIGHTS; i++)
	{
		vec3 positionToLight = normalize(vec3(lights[i].position) - fposition);
	
		vec3 ambient = vec3(0.3);

		float lDotN = max(0.0, dot(positionToLight, fnormal));
		finalColor += lights[i].intensity * lights[i].color * material.diffuse * lDotN; //this is super wrong but works for now?
	}


	

	FragColor = vec4(finalColor, 1.0);
	
	//FragColor = vec4(ourColor, 1.0);
}