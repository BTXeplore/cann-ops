#pragma once
#define REGISTER_TILING_DEFAULT(Type)
#define GET_TILING_DATA_WITH_STRUCT(Type, name, bytes) const auto name = *reinterpret_cast<const Type*>(bytes)
