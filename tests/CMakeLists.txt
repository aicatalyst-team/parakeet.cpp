function(pk_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE parakeet)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

pk_add_test(test_smoke)
pk_add_test(test_audio_io)
pk_add_test(test_model_loader)
pk_add_test(test_fft)
pk_add_test(test_mel)
pk_add_test(test_subsampling)
pk_add_test(test_relpos_attention)
set_tests_properties(test_model_loader test_mel test_subsampling test_relpos_attention
                     PROPERTIES LABELS "model")
# These tests read fixtures/baselines via paths relative to the project root.
set_tests_properties(test_mel test_subsampling test_relpos_attention
                     PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Python converter check (skips with exit 77 when the venv/model are absent).
find_program(PARAKEET_PY NAMES python3 HINTS ${CMAKE_SOURCE_DIR}/.venv/bin)
if(PARAKEET_PY)
  add_test(NAME check_convert COMMAND ${PARAKEET_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_convert.py)
  set_tests_properties(check_convert PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")

  add_test(NAME check_baseline COMMAND ${PARAKEET_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_baseline.py)
  set_tests_properties(check_baseline PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")
endif()
