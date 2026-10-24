@echo off
rem MSVC build. Run from any directory. Reuses a drive that already maps THIS worktree, else takes a free letter
rem (another agent may hold Q: / R:, so the letter is never assumed).
setlocal
set "WT=%~dp0..\..\.."
set "LLAMA=E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_baseline\llama.cpp"
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "NINJA_DIR=C:\Users\admin\AppData\Local\Programs\Python\Python311\Scripts"

if not exist "%WT%\tools\minimax\quant\CMakeLists.txt" (
  echo %WT% is not the quant worktree
  exit /b 1
)

rem A letter is ours when it already maps this worktree (tq_encode.hpp exists only here).
set "QDRV="
for %%D in (U V W X Y Z Q R S T) do (
  if not defined QDRV if exist %%D:\include\strata\kernels\cpu\tq_encode.hpp set "QDRV=%%D"
)
if not defined QDRV (
  for %%D in (U V W X Y Z Q R S T) do (
    if not defined QDRV if not exist %%D:\ (
      subst %%D: "%WT%" >nul 2>&1
      if exist %%D:\tools\minimax\quant\CMakeLists.txt set "QDRV=%%D"
    )
  )
)
set "RDRV="
for %%D in (U V W X Y Z Q R S T) do (
  if not defined RDRV if exist %%D:\ggml\include\ggml.h set "RDRV=%%D"
)
if not defined RDRV (
  for %%D in (U V W X Y Z Q R S T) do (
    if not defined RDRV if not exist %%D:\ (
      subst %%D: "%LLAMA%" >nul 2>&1
      if exist %%D:\ggml\include\ggml.h set "RDRV=%%D"
    )
  )
)
if not defined QDRV (echo no drive for the worktree & exit /b 1)
if not defined RDRV (echo no drive for llama.cpp & exit /b 1)
echo worktree %QDRV%:  llama.cpp %RDRV%:

call "%VCVARS%"
if errorlevel 1 exit /b 1
set "PATH=%NINJA_DIR%;%PATH%"

%QDRV%:
cd \tools\minimax\quant
if exist build-msvc\CMakeCache.txt (
  findstr /C:"CMAKE_C_COMPILER:FILEPATH" build-msvc\CMakeCache.txt | findstr /I "cl.exe" >nul
  if errorlevel 1 (
    echo removing non-MSVC build-msvc cache
    rmdir /s /q build-msvc
  )
)
cmake -S . -B build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_LLAMA_CPP_DIR=%RDRV%:/ -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe %*
if errorlevel 1 exit /b 1
cmake --build build-msvc
exit /b %ERRORLEVEL%
