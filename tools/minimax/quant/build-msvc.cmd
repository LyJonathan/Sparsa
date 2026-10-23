@echo off
rem MSVC build. Run from any directory. Maps Q: and R: if those letters are free.
rem Percent variables expand inside this file, after vcvars has changed PATH.
setlocal
set "WT=E:\Jon's Projects\AI\Strata Fork Minimax\wt-quant"
set "LLAMA=E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_baseline\llama.cpp"
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "NINJA_DIR=C:\Users\admin\AppData\Local\Programs\Python\Python311\Scripts"

subst Q: "%WT%" >nul 2>&1
subst R: "%LLAMA%" >nul 2>&1
if not exist Q:\tools\minimax\quant\CMakeLists.txt (
  echo Q: is not the quant worktree
  exit /b 1
)
if not exist R:\ggml\include\ggml.h (
  echo R: is not the llama.cpp tree
  exit /b 1
)

call "%VCVARS%"
if errorlevel 1 exit /b 1
set "PATH=%NINJA_DIR%;%PATH%"

Q:
cd \tools\minimax\quant
if exist build-msvc\CMakeCache.txt (
  findstr /C:"CMAKE_C_COMPILER:FILEPATH" build-msvc\CMakeCache.txt | findstr /I "cl.exe" >nul
  if errorlevel 1 (
    echo removing non-MSVC build-msvc cache
    rmdir /s /q build-msvc
  )
)
cmake -S . -B build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_LLAMA_CPP_DIR=R:/ -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe
if errorlevel 1 exit /b 1
cmake --build build-msvc
exit /b %ERRORLEVEL%
