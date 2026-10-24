@echo off
setlocal
set "LLAMA=E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_baseline\llama.cpp"
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "NINJA_DIR=C:\Users\admin\AppData\Local\Programs\Python\Python311\Scripts"
subst X: "E:\Jon's Projects\AI\Strata Fork Minimax\wt-trellis" >nul 2>&1
subst Y: "%LLAMA%" >nul 2>&1
call "%VCVARS%"
if errorlevel 1 exit /b 1
set "PATH=%NINJA_DIR%;%PATH%"
X:
cd \
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR=Y:/ -DSTRATA_MMQ_TQ_OVERLAY=OFF -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_CUDA_ARCHITECTURES=120
if errorlevel 1 exit /b 1
cmake --build build-cuda --target strata_minimax
exit /b %ERRORLEVEL%
