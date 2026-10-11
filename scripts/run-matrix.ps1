#!/usr/bin/env pwsh
# Multi-corpus / query-class A/B vs upstream. Total_matched + wall.
param(
  [string]$Opt = "D:/fff-target-keepers/release/fff-profile-driver.exe",
  [string]$Up = "C:/Users/kanal/Projects/fff-upstream/scratch/archive/bench-binaries/fff-profile-driver-upstream.exe",
  [int]$Iters = 8
)
$corpora = @(
  @{ n = "clanker-311"; p = "C:\Users\kanal\Projects\clanker-master" },
  @{ n = "fff-main-1.8k"; p = "C:\Users\kanal\Projects\fff-main" },
  @{ n = "F16-3.5k"; p = "C:\Users\kanal\Downloads\F-16" },
  @{ n = "ladybird-10k"; p = "C:\Users\kanal\Projects\ladybird" },
  @{ n = "sumo-227k"; p = "C:\Users\kanal\fff-corpus\sumo-worktree" }
)
$queries = @("score", "file_picker_", "main", "test", "file_picker.rs", "src", "handler")
foreach ($c in $corpora) {
  if (-not (Test-Path $c.p)) { continue }
  foreach ($q in $queries) {
    $om = (& $Opt $c.p $q $Iters 1 2>&1 | Out-String)
    $um = (& $Up  $c.p $q $Iters 1 2>&1 | Out-String)
    $oh = if ($om -match 'total_matched=(\d+)') { $Matches[1] } else { "?" }
    $ou = if ($om -match 'us_per_op=([0-9.]+)') { [math]::Round([double]$Matches[1]/1000,1) } else { 0 }
    $uu = if ($um -match 'us_per_op=([0-9.]+)') { [math]::Round([double]$Matches[1]/1000,1) } else { 0 }
    $rel = if ($uu -gt 0 -and $ou -gt 0) { [math]::Round($uu/$ou,2) } else { 0 }
    "{0,-14} {1,-14} opt={2,6}ms up={3,6}ms x={4} hits={5}" -f $c.n, $q, $ou, $uu, $rel, $oh
  }
}
