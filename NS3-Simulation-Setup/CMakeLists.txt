# Add Python support
find_package(PythonLibs 3.8 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Add your simulation target
add_executable(nano-network-sim
    nano-network-sim.cc
)

target_link_libraries(nano-network-sim
    ${PYTHON_LIBRARIES}
    ${NS3_LIBRARIES}
)