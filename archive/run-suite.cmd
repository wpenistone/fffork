@echo off
echo START %DATE% %TIME% > "C:\Users\kanal\Projects\fff-upstream\archive\paired-suite.lock"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\kanal\Projects\fff-upstream\archive\run_paired_sensors_c227k.ps1" >> "C:\Users\kanal\Projects\fff-upstream\archive\paired-sensors-c227k.console.log" 2>> "C:\Users\kanal\Projects\fff-upstream\archive\paired-sensors-c227k.err.log"
echo DONE %DATE% %TIME% >> "C:\Users\kanal\Projects\fff-upstream\archive\paired-suite.lock"
