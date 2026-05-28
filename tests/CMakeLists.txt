function(pk_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE parakeet)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

pk_add_test(test_smoke)
pk_add_test(test_audio_io)

# Python converter check (skips with exit 77 when the venv/model are absent).
find_program(PARAKEET_PY NAMES python3 HINTS ${CMAKE_SOURCE_DIR}/.venv/bin)
if(PARAKEET_PY)
  add_test(NAME check_convert COMMAND ${PARAKEET_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_convert.py)
  set_tests_properties(check_convert PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")
endif()
