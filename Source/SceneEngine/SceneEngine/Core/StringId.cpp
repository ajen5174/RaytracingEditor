#include "StringId.h"


uint32_t StringId::uniqueId = 0;
char* StringId::names = nullptr;

StringId::StringId(const char* string, bool unique)
{

	if (unique)
	{
		std::string uniqueString(string);
		uniqueString += std::to_string(uniqueId);
		uniqueId++;

		id = static_cast<uint32_t>(std::hash<std::string>{}(uniqueString.c_str()));
		index = id % MAX_ENTRIES;
		strcpy_s(names + (index * MAX_NAME_SIZE), MAX_NAME_SIZE, uniqueString.c_str());
	}
	else
	{
		id = static_cast<uint32_t>(std::hash<std::string>{}(string));
		index = id % MAX_ENTRIES;
		strcpy_s(names + (index * MAX_NAME_SIZE), MAX_NAME_SIZE, string);
	}
}

StringId::StringId(const std::string& string, bool unique)
	: StringId(string.c_str(), unique)
{
}

bool StringId::operator==(const StringId& other) const
{
	return (id == other.id);
}

bool StringId::operator!=(const StringId& other) const
{
	return (id != other.id);
}

bool StringId::operator<(const StringId& other) const
{
	return (id < other.id);
}

std::string StringId::ToString() const
{
	return std::string(cStr());
}

const char* StringId::cStr() const
{
	return names + (index * MAX_NAME_SIZE);
}

void StringId::AllocNames()
{
	names = new char[MAX_ENTRIES * MAX_NAME_SIZE];
}

void StringId::FreeNames()
{
	delete names;
}