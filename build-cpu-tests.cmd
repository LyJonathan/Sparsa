@echo off
rem Builds the CPU-side test targets of this worktree (tq_encode_parity, tq_cpu_parity) without CUDA.
setlocal
set "WT=%~dp0"
set "LLAMA=E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_baseline\llama.cpp"
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "NINJA_DIR=C:\Users\admin\AppData\Local\Programs\Python\Python311\Scripts"

subst X: "%WT%" >nul 2>&1
subst Y: "%LLAMA%" >nul 2>&1
if not exist X:\CMakeLists.txt exit /b 1
if not exist Y:\ggml\include\ggml.h exit /b 1

call "%VCVARS%"
if errorlevel 1 exit /b 1
set "PATH=%NINJA_DIR%;%PATH%"

X:
cd \
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_GGML_DIR=Y:/ ^
      -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe
if errorlevel 1 exit /b 1
cmake --build build-cpu --target tq_encode_parity tq_cpu_parity
exit /b %ERRORLEVEL%
