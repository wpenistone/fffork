$up = 'C:/Users/kanal/Projects/fff-upstream/.fff-bench/build/upstream/release/fff-profile-driver.exe'
$opt = 'C:/Users/kanal/Projects/fff-upstream/archive/bench-binaries/fff-profile-driver-keepers.exe'
$corpus = 'C:/Users/kanal/Projects/fff-upstream/.fff-bench/cache/sumo-worktree'
$out = 'C:/Users/kanal/Projects/fff-upstream/archive/paired-sensors.jsonl'
Remove-Item -LiteralPath $out -ErrorAction SilentlyContinue

$cases = @(
  @{q='file_picker_'; n=300; t=1},
  @{q='file_picker_'; n=300; t=4},
  @{q='score'; n=150; t=1},
  @{q='score'; n=150; t=4}
)

function Run-One($subject, $exe, $case, $block, $order) {
  $text = (& $exe $corpus $case.q $case.n $case.t 2>&1 | Out-String).Trim()
  Write-Host "$subject $($case.q) t=$($case.t): $text"
  if ($text -notmatch 'us_per_op=([0-9.]+)') { return }
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
  }
  ($rec | ConvertTo-Json -Compress) | Add-Content -LiteralPath $out
}

for ($b = 1; $b -le 3; $b++) {
  foreach ($case in $cases) {
    if ($b % 2 -eq 1) {
      Run-One 'upstream' $up $case $b 1
      Run-One 'keepers' $opt $case $b 2
    } else {
      Run-One 'keepers' $opt $case $b 1
      Run-One 'upstream' $up $case $b 2
    }
  }
}
Write-Host "WROTE $out"
