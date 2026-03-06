cmake_minimum_required(VERSION 3.19)
project(mlx_create_c C)

# Find the intended Python. You can pass -DPython3_EXECUTABLE=... at build time.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# ---- Python extension module (importable) ----
# Query Python for the full extension suffix (includes .so/.pyd and ABI)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')"
  OUTPUT_VARIABLE PY_EXT_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python EXT_SUFFIX: ${PY_EXT_SUFFIX}")

# --- Reusable C library with your logic ---
add_library(mlx_create_c_obj OBJECT src/mlx_create_c/mlx_create_c.c)
target_include_directories(mlx_create_c_obj PUBLIC src/mlx_create_c)
target_include_directories(mlx_create_c_obj PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(mlx_create_c_obj PUBLIC ${Python3_NumPy_INCLUDE_DIRS})

add_library(mlx_create_c STATIC $<TARGET_OBJECTS:mlx_create_c_obj>)
target_link_libraries(mlx_create_c PUBLIC m)
target_include_directories(mlx_create_c PUBLIC src/mlx_create_c)

# --- Python extension
add_library(_mlx_create_c SHARED src/mlx_create_c/_mlx_create_c.c)

# Use the exact filename Python expects, cross-platform
set_target_properties(_mlx_create_c PROPERTIES
  PREFIX ""
  OUTPUT_NAME "_mlx_create_c${PY_EXT_SUFFIX}" 
)

target_include_directories(_mlx_create_c PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(_mlx_create_c PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(_mlx_create_c PRIVATE Python3::Module)
target_link_libraries(_mlx_create_c PRIVATE Python3::NumPy)
target_link_libraries(_mlx_create_c PRIVATE mlx_create_c)

install(TARGETS _mlx_create_c
  LIBRARY DESTINATION mlx_create_c
  RUNTIME DESTINATION mlx_create_c
)

# No "lib" prefix and correct ABI suffix for Python modules
set_target_properties(_mlx_create_c PROPERTIES
  PREFIX ""
  SUFFIX "${Python3_EXTENSION_MODULE_SUFFIX}"
)

message(STATUS "Python ext suffix: ${Python3_EXTENSION_MODULE_SUFFIX}")

# Install the built module into the package
install(TARGETS _mlx_create_c LIBRARY DESTINATION mlx_create_c)

# # --- C tests via CTest ---
# include(CTest)
# if(BUILD_TESTING)
#   add_executable(test_mlx_create_c tests/c/test_mlx_create_c.c)
#   target_link_libraries(test_mlx_create_c PRIVATE mlx_create_c)
#   target_include_directories(test_mlx_create_c PRIVATE src/mlx_create_c)
#   add_test(NAME c_mlx_create_c COMMAND test_mlx_create_c)
# endif()
