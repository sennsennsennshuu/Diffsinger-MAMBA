@echo off
REM Build script for SSMOptimizer.dll
REM This script builds the C++ SSM optimizer library

echo ============================================
echo SSM Optimizer Build Script
echo ============================================

REM Check for Visual Studio
where cl >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Visual Studio compiler (cl) not found.
    echo Please run this script from a Visual Studio Developer Command Prompt.
    exit /b 1
)

REM Create build directory
if not exist build mkdir build
cd build

REM Set ONNX Runtime path (adjust as needed)
if not defined ONNXRUNTIME_ROOT (
    set "ONNXRUNTIME_ROOT=C:\Program Files\onnxruntime"
    echo ONNXRUNTIME_ROOT not set, using default: %ONNXRUNTIME_ROOT%
) else (
    echo Using ONNXRUNTIME_ROOT: %ONNXRUNTIME_ROOT%
)

REM Check if ONNX Runtime exists
if not exist "%ONNXRUNTIME_ROOT%\include\onnxruntime_cxx_api.h" (
    echo Warning: ONNX Runtime headers not found at %ONNXRUNTIME_ROOT%
    echo Building without ONNX Runtime support...
    set "BUILD_ONNX=OFF"
) else (
    set "BUILD_ONNX=ON"
)

REM Configure with CMake
echo.
echo Configuring with CMake...
if "%BUILD_ONNX%"=="ON" (
    cmake .. -G "Visual Studio 17 2022" -A x64 ^
        -DONNXRUNTIME_ROOT="%ONNXRUNTIME_ROOT%" ^
        -DSSM_USE_SIMD=ON ^
        -DSSM_USE_OPENMP=ON ^
        -DSSM_BUILD_TESTS=OFF
) else (
    cmake .. -G "Visual Studio 17 2022" -A x64 ^
        -DSSM_USE_SIMD=ON ^
        -DSSM_USE_OPENMP=ON ^
        -DSSM_BUILD_TESTS=OFF
)

if %errorlevel% neq 0 (
    echo Error: CMake configuration failed.
    exit /b 1
)

REM Build
echo.
echo Building SSMOptimizer.dll...
cmake --build . --config Release --parallel

if %errorlevel% neq 0 (
    echo Error: Build failed.
    exit /b 1
)

echo.
echo ============================================
echo Build completed successfully!
echo Output: build\Release\SSMOptimizer.dll
echo ============================================

REM Copy to OpenUtau Dependencies if requested
echo.
set /p COPY_DLL="Copy DLL to OpenUtau Dependencies? (Y/N): "
if /i "%COPY_DLL%"=="Y" (
    set "DEST_DIR=C:\Users\Asus\Documents\OpenUtau\Dependencies\SSM"
    if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"
    copy "Release\SSMOptimizer.dll" "%DEST_DIR%\" >nul
    copy "..\ssm_optimizer.h" "%DEST_DIR%\" >nul
    echo Copied to: %DEST_DIR%
)

cd ..
echo.
echo Done!
pause
