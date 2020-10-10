#pragma once

#include <string>

class StringId
{
public:
	StringId() : id(0), index(0) {}
	StringId(const char* string, bool unique = false);

	bool operator==(const StringId& other) const;
	bool operator!=(const StringId& other) const;
	bool operator<(const StringId& other) const;

	std::string ToString() const;
	const char* cStr() const;

	uint64_t GetId() const { return id; }

	static void AllocNames();
	static void FreeNames();


private:
	uint64_t id;
	uint64_t index;
	static const uint64_t MAX_ENTRIES = 65536;
	static const uint64_t MAX_NAME_SIZE = 64;

	static uint64_t uniqueId;
	static char* names;

};