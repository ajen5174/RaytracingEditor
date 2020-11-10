#pragma once
#include "StringId.h"
#include "../external/rapidjson/document.h"

#ifndef OBJECT_H
#define OBJECT_H
class Object
{
public:
	Object(const StringId& name) : name(name) {}
	virtual void Destroy() = 0;
	virtual bool Load(const rapidjson::Value&) = 0;
	virtual void BuildJSON(rapidjson::Value& v, rapidjson::MemoryPoolAllocator<>& mem) = 0;
	virtual void Initialize() = 0;
	
	StringId& GetName() { return name; };
	void SetName(char* newName)
	{
		name = newName;
	}

protected:
	StringId name;
};
#endif