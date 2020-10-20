#pragma once
#include <glad\glad.h>

class PickTexture
{
public:
	struct PickInfo
	{
		float objectID;
		float drawID;
		float primitiveID;

		PickInfo()
		{
			objectID = 0.0f;
			drawID = 0.0f;
			primitiveID = 0.0f;
		}
	};

	PickTexture();
	~PickTexture();

	bool Initialize(int windowHeight, int windowWidth);

	void EnableWriting();

	void DisableWriting();
	
	PickInfo ReadPixel(int x, int y);


private:
	GLuint fbo;
	GLuint pickTexture;
	GLuint depthTexture;
};