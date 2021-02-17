
cmake_minimum_required(VERSION 3.5)

project(sampleTRT)

if (UNIX)
    set(RUNTIME_LIB pthread dl rt)
    set(RUNTIME_LIB_TRT ${RUNTIME_LIB})
endif()

set (ARCH armv8)

set (TRTVER 6)
set (CUDAVER 100)

# set (TRTVER 7)
# set (CUDAVER 102)

# set (LINKAGE static)
set(LINKAGE dynamic)

add_library(sampleTRTLib SHARED sampleTRTLib.cpp)
target_compile_options(sampleTRTLib PRIVATE -v)
set_property(TARGET sampleTRTLib PROPERTY CXX_STANDARD 17)
target_include_directories(sampleTRTLib PRIVATE tensorrt${TRTVER}/${ARCH}/include cuda${CUDAVER}/${ARCH}/include protobuf/${ARCH}/include)
# if (LINUX)
#     target_link_options(sampleTRTLib PRIVATE -Wl,-undefined,dynamic_lookup; -flto)
# endif()
target_link_options(sampleTRTLib PRIVATE -v)
target_link_directories(sampleTRTLib PRIVATE tensorrt${TRTVER}/${ARCH}/lib cuda${CUDAVER}/${ARCH}/lib protobuf/${ARCH}/lib )
if ( "${LINKAGE}" STREQUAL "static" )
    target_link_libraries(sampleTRTLib nvcaffe_parser.a nvinfer_plugin_static nvinfer_static nvparsers_static protobuf cudart_static ${RUNTIME_LIB_TRT})
else()
    target_link_libraries(sampleTRTLib nvcaffe_parser nvinfer_plugin nvinfer nvparsers protobuf cudart_static ${RUNTIME_LIB_TRT})
endif()


add_executable(sampleTRT sampleTRT.cpp)
set_property(TARGET sampleTRT PROPERTY CXX_STANDARD 17)
target_link_libraries(sampleTRT ${RUNTIME_LIB})
target_link_options(sampleTRT PRIVATE  -Wl,--no-undefined)

install(TARGETS sampleTRT sampleTRTLib DESTINATION bin${TRTVER}-${LINKAGE})
install(DIRECTORY data DESTINATION bin${TRTVER}-${LINKAGE})
if ( "${LINKAGE}" STREQUAL "dynamic" )
    install(DIRECTORY  tensorrt${TRTVER}/${ARCH}/lib/ DESTINATION bin${TRTVER}-${LINKAGE} FILES_MATCHING PATTERN "*.so*")
    install(DIRECTORY  cuda${CUDAVER}/${ARCH}/lib/ DESTINATION bin${TRTVER}-${LINKAGE} FILES_MATCHING PATTERN "*.so*")
endif()



