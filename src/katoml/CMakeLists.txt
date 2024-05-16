add_subdirectory(mlcompiler)
add_subdirectory(mlsupport)
add_subdirectory(mltensor)
add_subdirectory(mlapp)

pybind11_add_module(katoml pybind.cpp)
target_link_libraries(katoml PUBLIC mlcompiler mltensor)

install(TARGETS katoml 
  COMPONENT python
  LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
  )
