#pragma once

#include <string>

#ifndef STRINGID_H
#define STRINGID_H
class StringId
{
public:
	StringId() : id(0), index(0) {}
	StringId(const char* string, bool unique = false);
	StringId(const std::string& string, bool unique = false);

	bool operator==(const StringId& other) const;
	bool operator!=(const StringId& other) const;
	bool operator<(const StringId& other) const;

	std::string ToString() const;
	const char* cStr() const;

	uint32_t GetId() const { return id; }
	float GetFloatId() const { return static_cast<float>(id); }

	static void AllocNames();
	static void FreeNames();


private:
	uint32_t id;
	uint32_t index;
	static const uint32_t MAX_ENTRIES = 65536;
	static const uint32_t MAX_NAME_SIZE = 256;

	static uint32_t uniqueId;
	static char* names;

};
#endif