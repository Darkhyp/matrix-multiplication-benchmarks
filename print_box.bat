@echo off
setlocal EnableDelayedExpansion

set "text=%~1"
set /a text_width=0

:count_chars
if not "!text:~%text_width%,1!"=="" (
    set /a text_width+=1
    goto count_chars
)
set /a text_width+=2

set /a width=80
if !text_width! GTR !width! set /a width=text_width

if not "%~2"=="" (
    set /a requested=%~2
    if !requested! GTR !width! set /a width=requested
)

set "line="
for /l %%i in (1,1,!width!) do set "line=!line!─"

set /a padding=width-text_width
set "spaces="
for /l %%i in (1,1,!padding!) do set "spaces=!spaces! "

echo ┌!line!┐
echo │ !text!!spaces! │
echo └!line!┘

endlocal
exit /b
