cmake_minimum_required(VERSION 3.20)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

# ============================================================================
# General
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(BINDER_CPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/BINDER/src/cpu")
set(BINDER_GPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/BINDER/src/gpu")


# ============================================================================
# Backend selection
#
# Default:
#   CUDA available -> GPU build
#   CUDA unavailable -> CPU build
#
# Override:
#   -DBINDER_FORCE_CPU=ON
# ============================================================================

option(
    BINDER_FORCE_CPU
    "Force CPU backend even when CUDA is available"
    OFF
)

option(
    BINDER_DISABLE_TOPOLOGY_CORRECTION
    "Exclude the topology correction algorithm (derivative of FSL warpfns, non-commercial FSL license) from the build"
    OFF
)

set(BINDER_BUILD_GPU OFF)

if(NOT BINDER_FORCE_CPU)

    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        set(BINDER_BUILD_GPU ON)
    endif()

endif()


# ============================================================================
# GPU BACKEND
# ============================================================================

if(BINDER_BUILD_GPU)

    message(STATUS "")
    message(STATUS "========================================")
    message(STATUS " BINDER backend: CUDA")
    message(STATUS " CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "========================================")
    message(STATUS "")

    enable_language(CUDA)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
    endif()
    
    find_package(CUDAToolkit REQUIRED)

    set(
        BINDER_GPU_SOURCES

        # Pybind11 interface
        "${BINDER_GPU_DIR}/bindings.cpp"

        # Existing CUDA implementation
        "${BINDER_GPU_DIR}/DCT3D.cu"
        "${BINDER_GPU_DIR}/Registration.cu"
        "${BINDER_GPU_DIR}/NonLinearTransformation.cu"
        "${BINDER_GPU_DIR}/FFT.cu"
    )

    pybind11_add_module(
        _core
        MODULE
        NO_EXTRAS
        ${BINDER_GPU_SOURCES}
    )

    target_include_directories(
        _core
        PRIVATE
            "${BINDER_GPU_DIR}"
    )

    target_link_libraries(
        _core
        PRIVATE
            CUDA::cudart
            CUDA::curand
            CUDA::cublas
            CUDA::cufft
    )

    set_target_properties(
        _core
        PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON

            # Compile CUDA translation units separately
            CUDA_SEPARABLE_COMPILATION ON

            # IMPORTANT:
            # Perform the CUDA device-link step for this Python module.
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_compile_options(
        _core
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    )


# ============================================================================
# CPU BACKEND
# ============================================================================

else()

    message(STATUS "")
    message(STATUS "========================================")

    if(BINDER_FORCE_CPU)
        message(STATUS " BINDER backend: CPU (forced)")
    else()
        message(STATUS " BINDER backend: CPU (CUDA not available)")
    endif()

    message(STATUS "========================================")
    message(STATUS "")


    # ========================================================================
    # Eigen
    # ========================================================================

    find_package(Eigen3 3.3 CONFIG QUIET)

    # Fallback for systems where Eigen's CMake config isn't installed
    if(NOT TARGET Eigen3::Eigen)

        find_path(
            EIGEN3_INCLUDE_DIR
            NAMES Eigen/Core
            PATH_SUFFIXES eigen3
        )

        if(NOT EIGEN3_INCLUDE_DIR)
            message(
                FATAL_ERROR
                "Eigen3 could not be found.\n"
                "Install Eigen3 or set CMAKE_PREFIX_PATH."
            )
        endif()

        add_library(Eigen3::Eigen INTERFACE IMPORTED)

        set_target_properties(
            Eigen3::Eigen
            PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )

    endif()


    # ========================================================================
    # FFTW
    # ========================================================================

    #
    # First try FFTW's CMake config.
    #
    find_package(FFTW3 CONFIG QUIET)

    if(TARGET FFTW3::fftw3)

        set(BINDER_FFTW_TARGET FFTW3::fftw3)

    else()

        #
        # Optional pkg-config hints.
        #
        find_package(PkgConfig QUIET)

        if(PkgConfig_FOUND)
            pkg_check_modules(PC_FFTW3 QUIET fftw3)
        endif()

        find_path(
            FFTW3_INCLUDE_DIR
            NAMES fftw3.h
            HINTS
                ${PC_FFTW3_INCLUDE_DIRS}
                ${FFTW3_ROOT}
                $ENV{FFTW3_ROOT}
            PATH_SUFFIXES
                include
        )

        find_library(
            FFTW3_LIBRARY
            NAMES fftw3
            HINTS
                ${PC_FFTW3_LIBRARY_DIRS}
                ${FFTW3_ROOT}
                $ENV{FFTW3_ROOT}
            PATH_SUFFIXES
                lib
                lib64
        )

        if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARY)

            message(
                FATAL_ERROR
                "FFTW3 could not be found.\n"
                "Install FFTW3 or set FFTW3_ROOT / CMAKE_PREFIX_PATH."
            )

        endif()

        add_library(BINDER_FFTW3 UNKNOWN IMPORTED)

        set_target_properties(
            BINDER_FFTW3
            PROPERTIES
                IMPORTED_LOCATION "${FFTW3_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}"
        )

        set(BINDER_FFTW_TARGET BINDER_FFTW3)

    endif()


    # ========================================================================
    # FFTW threads
    #
    # On some platforms threading is compiled into the main fftw3 library and
    # there is no separate fftw3_threads. Look for a separate one; if absent,
    # link nothing extra.
    # ========================================================================

    if(TARGET FFTW3::fftw3_threads)

        set(
            BINDER_FFTW_THREADS_TARGET
            FFTW3::fftw3_threads
        )

    else()

        find_library(
            FFTW3_THREADS_LIBRARY
            NAMES
                fftw3_threads
            HINTS
                ${PC_FFTW3_LIBRARY_DIRS}
                ${FFTW3_ROOT}
                $ENV{FFTW3_ROOT}
            PATH_SUFFIXES
                lib
                lib64
        )

        if(FFTW3_THREADS_LIBRARY)

            add_library(
                BINDER_FFTW3_THREADS
                UNKNOWN
                IMPORTED
            )

            set_target_properties(
                BINDER_FFTW3_THREADS
                PROPERTIES
                    IMPORTED_LOCATION "${FFTW3_THREADS_LIBRARY}"
            )

            set(
                BINDER_FFTW_THREADS_TARGET
                BINDER_FFTW3_THREADS
            )

        else()

            message(STATUS "fftw3_threads not found separately; assuming it is in the main FFTW library.")
            set(BINDER_FFTW_THREADS_TARGET "")

        endif()

    endif()


    # ========================================================================
    # FFTW license notice
    #
    # FFTW is distributed under the GPL. Linking against it makes the built
    # _core extension (and anything linked against it) subject to the GPL.
    # See http://www.fftw.org/ for the commercial license option from MIT.
    # ========================================================================

    set(msg "ATTENTION: This build links against FFTW.")
    set(msg "${msg} FFTW is distributed under a GPL license.")
    set(msg "${msg} By linking against it, the _core extension that is built")
    set(msg "${msg} will be covered by a GPL license, and so will anything")
    set(msg "${msg} linked against it.")
    message(WARNING "${msg}")


    # ========================================================================
    # Threads / OpenMP
    # ========================================================================

    find_package(Threads REQUIRED)
    find_package(OpenMP REQUIRED COMPONENTS CXX)


    # ========================================================================
    # CPU sources
    # ========================================================================

    set(
        BINDER_CPU_SOURCES

        # Pybind11 interface
        "${BINDER_CPU_DIR}/bindings.cpp"

        "${BINDER_CPU_DIR}/Registration.cpp"
        "${BINDER_CPU_DIR}/Spline.cpp"

        "${BINDER_CPU_DIR}/Likelihood.cpp"
        "${BINDER_CPU_DIR}/MILikelihood.cpp"
        "${BINDER_CPU_DIR}/SSDLikelihood.cpp"

        "${BINDER_CPU_DIR}/MILikelihoodFilter.cpp"
        "${BINDER_CPU_DIR}/SSDLikelihoodFilter.cpp"

        "${BINDER_CPU_DIR}/Transformation.cpp"
        "${BINDER_CPU_DIR}/NonLinearTransformation.cpp"
        "${BINDER_CPU_DIR}/AffineTransformation.cpp"
        "${BINDER_CPU_DIR}/RigidTransformation.cpp"

        "${BINDER_CPU_DIR}/DCT.cpp"
        "${BINDER_CPU_DIR}/FFT.cpp"

        "${BINDER_CPU_DIR}/Smoothing_Spline.cpp"
        "${BINDER_CPU_DIR}/utils.cpp"
    )

    #
    # Do NOT include main.cpp here.
    # A Python extension doesn't have main().
    #

    pybind11_add_module(
        _core
        MODULE
        ${BINDER_CPU_SOURCES}
    )

    target_include_directories(
        _core
        PRIVATE
            "${BINDER_CPU_DIR}"
    )

    target_link_libraries(
        _core
        PRIVATE
            Eigen3::Eigen
            ${BINDER_FFTW_TARGET}
            ${BINDER_FFTW_THREADS_TARGET}
            Threads::Threads
            OpenMP::OpenMP_CXX
    )

    target_compile_features(
        _core
        PRIVATE
            cxx_std_17
    )

endif()

# ============================================================================
# Topology correction (optional)
#
# The topology correction algorithm is a derivative of FSL's warpfns
# (https://git.fmrib.ox.ac.uk/fsl/warpfns/-/blob/master/warpfns.cc),
# which is distributed under the FSL license:
#   https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html
#
# Built by default. Disable with -DBINDER_DISABLE_TOPOLOGY_CORRECTION=ON.
# ============================================================================

if(NOT BINDER_DISABLE_TOPOLOGY_CORRECTION)

    if(BINDER_BUILD_GPU)
        target_sources(
            _core
            PRIVATE
                "${BINDER_GPU_DIR}/topology_correction_3D.cu"
        )
    else()
        target_sources(
            _core
            PRIVATE
                "${BINDER_CPU_DIR}/topology_correction_2D.cpp"
                "${BINDER_CPU_DIR}/topology_correction_3D.cpp"
        )
    endif()

    target_compile_definitions(
        _core
        PRIVATE
            BINDER_WITH_TOPOLOGY_CORRECTION
    )

    set(msg "ATTENTION: This build includes the topology correction algorithm,")
    set(msg "${msg} which is a derivative of FSL's warpfns")
    set(msg "${msg} (https://git.fmrib.ox.ac.uk/fsl/warpfns).")
    set(msg "${msg} FSL is distributed under a license that permits NON-COMMERCIAL")
    set(msg "${msg} use only: any use for which a financial return is received")
    set(msg "${msg} (including selling or licensing products/services derived from")
    set(msg "${msg} it) requires a commercial license from Oxford University")
    set(msg "${msg} Innovation (fsl@innovation.ox.ac.uk). This restriction applies")
    set(msg "${msg} to the _core extension and anything linked against it.")
    set(msg "${msg} See https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html.")
    set(msg "${msg} Disable with -DBINDER_DISABLE_TOPOLOGY_CORRECTION=ON.")
    message(WARNING "${msg}")

else()

    message(STATUS "BINDER: topology correction excluded (BINDER_DISABLE_TOPOLOGY_CORRECTION=ON).")

endif()

# ============================================================================
# Install
# ============================================================================

install(
    TARGETS _core
    LIBRARY DESTINATION BINDER
    RUNTIME DESTINATION BINDER
)
