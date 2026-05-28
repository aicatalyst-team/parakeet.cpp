add_executable(parakeet-cli main.cpp)
target_link_libraries(parakeet-cli PRIVATE parakeet)
target_include_directories(parakeet-cli PRIVATE ${CMAKE_SOURCE_DIR}/src)
