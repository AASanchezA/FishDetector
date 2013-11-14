# -----------------------------------------------
# BUILD THE FISH DETECTOR FROM SOURCE
# -----------------------------------------------

set(LIBS        ${Boost_LIBRARIES}
                ${OpenCV_LIBS}
                ${PartsBasedDetector_LIBRARIES}
)

# with cvmatio support
if (WITH_CVMATIO)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_MATLABIO")
endif()

# as an executable
if (BUILD_EXECUTABLE)
    set(SRC_FILES FishDetector.cpp)
    add_executable(${PROJECT_NAME}_bin ${SRC_FILES})
    target_link_libraries(${PROJECT_NAME}_bin ${LIBS})# ${PROJECT_NAME})
    set_target_properties(${PROJECT_NAME}_bin PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
    install(TARGETS ${PROJECT_NAME}_bin
            RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
    )
endif()
