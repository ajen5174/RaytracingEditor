#pragma once
#include <string>
#include <rapidjson/rapidjson.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/document.h>
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
		auto iter = value.FindMember("translation");
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
}