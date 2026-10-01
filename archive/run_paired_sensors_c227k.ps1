# July-style paired process runs with full cost sensors.
# Corpus: clean 227k sumo worktree OUTSIDE any git tree.
# 6 counterbalanced blocks. Warm index. Lower is better (except QPS).

$up = 'C:/Users/kanal/Projects/fff-upstream/archive/bench-binaries/fff-profile-driver-upstream.exe'
$opt = 'C:/Users/kanal/Projects/fff-upstream/archive/bench-binaries/fff-profile-driver-optimized.exe'
$corpus = 'C:/Users/kanal/fff-corpus/sumo-worktree'
$out = 'C:/Users/kanal/Projects/fff-upstream/archive/paired-sensors-c227k.jsonl'
$log = 'C:/Users/kanal/Projects/fff-upstream/archive/paired-sensors-c227k.log'
Remove-Item -LiteralPath $out -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $log -ErrorAction SilentlyContinue

if (-not (Test-Path $corpus)) { throw "corpus missing: $corpus" }
if (-not (Test-Path "$corpus/.git")) { Write-Host "OK: corpus has no .git" } else { throw "corpus has .git - aborting" }

$cases = @(
  @{q='file_picker_'; n=150; t=1},
  @{q='file_picker_'; n=150; t=4},
  @{q='score'; n=100; t=1},
  @{q='score'; n=100; t=4}
)

function Run-One($subject, $exe, $case, $block, $order) {
  $text = (& $exe $corpus $case.q $case.n $case.t 2>&1 | Out-String).Trim()
  $line = "block=$block order=$order $subject $($case.q) t=$($case.t): $text"
  Write-Host $line
  Add-Content -LiteralPath $log -Value $line
  if ($text -notmatch 'us_per_op=([0-9.]+)') {
    Add-Content -LiteralPath $log -Value "FAIL parse $subject $($case.q) t=$($case.t)"
    return
  }
  $m = @{}
  foreach ($pair in ($text -split '\s+')) {
    if ($pair -match '^(\w+)=(.+)$') { $m[$matches[1]] = $matches[2] }
  }
  $rec = [ordered]@{
    subject=$subject; block=$block; order=$order
    query=$case.q; threads=[int]$case.t; iterations=[int]$case.n
    us_per_op=[double]$m['us_per_op']
    cpu_us_per_op=[double]$m['cpu_us_per_op']
    cycles_per_op=[double]$m['cycles_per_op']
    read_b_op=[double]$m['read_b_op']
    faults_per_100k=[double]$m['faults_per_100k']
    peak_rss_mb=[double]$m['peak_rss_mb']
    checksum=[int64]$m['checksum']
    corpus='sumo-worktree-227k'
  }
  ($rec | ConvertTo-Json -Compress) | Add-Content -LiteralPath $out
}

for ($b = 1; $b -le 6; $b++) {
  foreach ($case in $cases) {
    if ($b % 2 -eq 1) {
      Run-One 'upstream' $up $case $b 1
      Run-One 'optimized' $opt $case $b 2
    } else {
      Run-One 'optimized' $opt $case $b 1
      Run-One 'upstream' $up $case $b 2
    }
  }
}
Write-Host "WROTE $out"
Add-Content -LiteralPath $log -Value "WROTE $out"

