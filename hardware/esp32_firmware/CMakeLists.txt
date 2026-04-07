cmake_minimum_required(VERSION 3.16)
# Border Sentinel — ESP32 Firmware Build
# Build: idf.py build
# Flash: idf.py -p /dev/ttyUSB0 flash monitor
set(EXTRA_COMPONENT_DIRS 
    "${CMAKE_CURRENT_LIST_DIR}/components/tflite-micro"
    "${CMAKE_CURRENT_LIST_DIR}/components/esp-dsp"
)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(border_sentinel_node VERSION 2.0.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 11)
add_compile_options(-O2 -ffunction-sections -fdata-sections)
