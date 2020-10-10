#pragma once
#include "StringId.h"
#include "../external/rapidjson/document.h"


class Object
{
public:
	Object(const StringId& name) : name(name) {}
	virtual void Destroy() = 0;
	virtual bool Load(const rapidjson::Value&) = 0;
	virtual void Initialize() = 0;
	
	StringId& GetName() { return name; };


protected:
	StringId name;
};