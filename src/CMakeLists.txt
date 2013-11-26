# -----------------------------------------------
# BUILD THE FISH DETECTOR FROM SOURCE
# -----------------------------------------------

set(SRC_FILES   ConfigFile.cpp 
                functions.cpp
)

set(LIBS        ${Boost_LIBRARIES}
                ${OpenCV_LIBS}
                ${PartsBasedDetector_LIBRARIES}
)

# with cvmatio support
if (WITH_CVMATIO)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_MATLABIO")
endif()

# as a library (always)
add_library(${PROJECT_NAME}_lib SHARED ${SRC_FILES})
target_link_libraries(${PROJECT_NAME}_lib ${LIBS})
set_target_properties(${PROJECT_NAME}_lib PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME}_lib 
        DESTINATION ${PROJECT_SOURCE_DIR}/lib
)

# as an executable
if (BUILD_EXECUTABLE)
    set(SRC_FILES FishDetector.cpp)
    add_executable(${PROJECT_NAME}_bin ${SRC_FILES})
    target_link_libraries(${PROJECT_NAME}_bin ${LIBS} ${PROJECT_NAME}_lib)
    set_target_properties(${PROJECT_NAME}_bin PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
    install(TARGETS ${PROJECT_NAME}_bin
            RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
    )
endif()
