#pragma once
#include <glad\glad.h>

class PickTexture
{
public:
	PickTexture();
	~PickTexture();

	bool Initialize(int windowHeight, int windowWidth);

	void EnableWriting();

	void DisableWriting();

	float ReadPixel(int x, int y);

private:
	GLuint fbo;
	GLuint pickTexture;
	GLuint depthTexture;
};