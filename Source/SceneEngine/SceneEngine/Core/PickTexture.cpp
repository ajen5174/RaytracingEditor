#include "PickTexture.h"
#include <cstddef>

PickTexture::PickTexture()
{
}

PickTexture::~PickTexture()
{
}

bool PickTexture::Initialize(int windowHeight, int windowWidth)
{
    //Create buffer
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    //Create texture to hold our pixel info
    glGenTextures(1, &pickTexture);
    glBindTexture(GL_TEXTURE_2D, pickTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, windowWidth, windowHeight, 0, GL_RGB, GL_FLOAT, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
        pickTexture, 0);

    //Create depth texture to be used to guarantee the closest object gets selected.
    glGenTextures(1, &depthTexture);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, windowWidth, windowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);

    glReadBuffer(GL_NONE);

    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    GLenum Status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

    if (Status != GL_FRAMEBUFFER_COMPLETE) {
        return false;
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return true;
}

void PickTexture::EnableWriting()
{
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
}

void PickTexture::DisableWriting()
{
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

float PickTexture::ReadPixel(int x, int y)
{
    //bind the buffer to READ
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    float pixel[3] = { 0.0f };
    //this lets us read the pixels
    glReadPixels(x, y, 1, 1, GL_RGB, GL_FLOAT, &pixel);

    //unbind
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

    return pixel;
}
