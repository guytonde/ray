set(GLM_VERSION 0.9.9.0)
find_package(GLM ${GLM_VERSION} QUIET EXACT)

if(glm_FOUND)
	include_directories(${glm_INCLUDE_DIR})
	message(STATUS "Using system glm from package config: ${glm_INCLUDE_DIR}")
else()
	find_path(GLM_INCLUDE_DIR glm/glm.hpp
		PATHS
			/usr/include
			/usr/local/include
			${CMAKE_SOURCE_DIR}/third-party/glm
		NO_DEFAULT_PATH
	)

	if(GLM_INCLUDE_DIR)
		include_directories(BEFORE SYSTEM ${GLM_INCLUDE_DIR})
		message(STATUS "Using glm include path: ${GLM_INCLUDE_DIR}")
	else()
		message(FATAL_ERROR
			"GLM headers were not found. Install system glm or initialize third-party/glm.")
	endif()
endif()

add_definitions(-DGLM_ENABLE_EXPERIMENTAL -DGLM_FORCE_SIZE_FUNC=1 -DGLM_FORCE_RADIANS=1)

# vim: tw=78
