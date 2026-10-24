@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 "%~dp0io_probe.cpp" /Fe:"%~dp0io_probe.exe"
