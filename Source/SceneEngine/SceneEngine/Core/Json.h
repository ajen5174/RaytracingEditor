#pragma once
#include <string>
#include <rapidjson/rapidjson.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/document.h>
#include <rapidjson/allocators.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/filewritestream.h>
#include <glm/glm/glm.hpp>
#include <fstream>
#include "StringId.h"



namespace json
{
	inline bool LoadFromFile(std::string path, rapidjson::Document& document)
	{
		std::ifstream file(path);
		rapidjson::IStreamWrapper isw(file);
		document.ParseStream(isw);
		if (document.IsObject())
		{
			return true;
		}
		return false;
	}

	inline bool GetInt(const rapidjson::Value& value, const char* property, int& data)
	{
		auto iter = value.FindMember(property);
		if (iter == value.MemberEnd())
		{
			return false;
		}

		auto& result = iter->value;
		if (result.IsInt() == false)
		{
			return false;
		}

		data = result.GetInt();
		return true;
	}

	inline bool GetFloat(const rapidjson::Value& value, const char* property, float& data)
	{
		auto iter = value.FindMember(property);
		if (iter == value.MemberEnd())
		{
			return false;
		}

		auto& result = iter->value;
		if (result.IsDouble() == false)
		{
			return false;
		}

		data = result.GetFloat();
		return true;
	}

	inline bool GetString(const rapidjson::Value& value, const char* property, std::string& data)
	{
		auto iter = value.FindMember(property);
		if (iter == value.MemberEnd())
		{
			return false;
		}

		auto& result = iter->value;
		if (result.IsString() == false)
		{
			return false;
		}

		data = result.GetString();
		return true;
	}

	inline bool GetBool(const rapidjson::Value& value, const char* property, bool& data)
	{
		auto iter = value.FindMember(property);
		if (iter == value.MemberEnd())
		{
			return false;
		}

		auto& result = iter->value;
		if (result.IsBool() == false)
		{
			return false;
		}

		data = result.GetBool();
		return true;
	}

	inline bool GetVec3(const rapidjson::Value& value, const char* property, glm::vec3& data)
	{
		auto iter = value.FindMember(property);
		if (iter == value.MemberEnd())
		{
			return false;
		}

		auto& result = iter->value;
		if (result.IsArray() == false || result.Size() != 3)
		{
			return false;
		}

		for (rapidjson::SizeType i = 0; i < 3; i++)
		{
			if (result[i].IsDouble() == false)
			{
				return false;
			}
		}
		data = glm::vec3(result[0].GetFloat(), result[1].GetFloat(), result[2].GetFloat());

		return true;
	}

	inline bool GetName(const rapidjson::Value& value, const char* property, StringId& data)
	{
		auto iter = value.FindMember(property);
		if (iter == value.MemberEnd())
		{
			return false;
		}

		auto& result = iter->value;
		if (result.IsString() == false)
		{
			return false;
		}

		data = result.GetString();
		return true;
	}



	inline void BuildVec3(rapidjson::Value& value, const char* property, glm::vec3 data, rapidjson::MemoryPoolAllocator<>& mem)
	{
		rapidjson::Value key;
		key.SetString(property, strlen(property), mem); //potentially need to +1 here
		rapidjson::Value vecValue;
		vecValue.SetArray();
		vecValue.PushBack(data.x, mem);
		vecValue.PushBack(data.y, mem);
		vecValue.PushBack(data.z, mem);
		value.AddMember(key, vecValue, mem);
	}

	inline void BuildName(rapidjson::Value& value, const char* property, StringId& name, rapidjson::MemoryPoolAllocator<>& mem)
	{
		rapidjson::Value nameKey(property, strlen(property), mem);
		rapidjson::Value nameValue;
		nameValue.SetString(name.cStr(), strlen(name.cStr()));
		value.AddMember(nameKey, nameValue, mem);
	}

	
	inline void BuildCString(rapidjson::Value& value, const char* property, const char* str, rapidjson::MemoryPoolAllocator<>& mem)
	{
		rapidjson::Value key;
		key.SetString(property, strlen(property), mem);
		rapidjson::Value stringValue(str, strlen(str), mem);
		value.AddMember(key, stringValue, mem);
	}

	inline void BuildString(rapidjson::Value& value, const char* property, std::string str, rapidjson::MemoryPoolAllocator<>& mem)
	{
		BuildCString(value, property, str.c_str(), mem);
	}

	inline void BuildBool(rapidjson::Value& value, const char* property, bool b, rapidjson::MemoryPoolAllocator<>& mem)
	{
		rapidjson::Value key(property, strlen(property), mem);
		value.AddMember(key, b, mem);
	}

	inline void BuildInt(rapidjson::Value& value, const char* property, int i, rapidjson::MemoryPoolAllocator<>& mem)
	{
		rapidjson::Value key(property, strlen(property), mem);
		value.AddMember(key, i, mem);
	}

	inline void BuildFloat(rapidjson::Value& value, const char* property, float f, rapidjson::MemoryPoolAllocator<>& mem)
	{
		rapidjson::Value key(property, strlen(property), mem);
		value.AddMember(key, f, mem);
	}
}