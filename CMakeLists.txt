cmake_minimum_required( VERSION 3.0 )
project( orb_mono_vo )

set( CMAKE_BUILD_TYPE "Debug" )

find_package( OpenCV 3 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} 
                     "./include" )

add_executable( orb_vo 
./src/VisualOdometry.cpp
./src/main.cpp)

target_link_libraries( orb_vo ${OpenCV_LIBS} )