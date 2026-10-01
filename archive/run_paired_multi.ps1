# Multi-corpus paired sensors: optimized vs upstream
# Shapes: sumo 227k, linux 100k, pager-flat 51k, faces 19k
# ASCII only.

$up = 'C:/Users/kanal/Projects/fff-upstream/archive/bench-binaries/fff-profile-driver-upstream.exe'
$opt = 'C:/Users/kanal/Projects/fff-upstream/archive/bench-binaries/fff-profile-driver-optimized.exe'
$out = 'C:/Users/kanal/Projects/fff-upstream/archive/paired-sensors-multi.jsonl'
$log = 'C:/Users/kanal/Projects/fff-upstream/archive/paired-sensors-multi.log'
Remove-Item -LiteralPath $out, $log -ErrorAction SilentlyContinue

$corpora = @(
  @{id='sumo-227k'; path='C:/Users/kanal/fff-corpus/sumo-worktree'; n_fp=100; n_sc=60},
  @{id='linux-100k'; path='C:/Users/kanal/Documents/antigravity/keen-oppenheimer'; n_fp=80; n_sc=50},
  @{id='pager-flat-51k'; path='C:/Users/kanal/Projects/911-pager-v2'; n_fp=80; n_sc=50},
  @{id='faces-19k'; path='C:/Users/kanal/Projects/studyofus-faces'; n_fp=60; n_sc=40}
)
$cases = @(
  @{q='file_picker_'; t=1; key='n_fp'},
  @{q='file_picker_'; t=4; key='n_fp'},
  @{q='score'; t=1; key='n_sc'},
  @{q='score'; t=4; key='n_sc'}
)

function Run-One($subject, $exe, $corpusId, $corpus, $case, $block, $order, $iters) {
  $text = (& $exe $corpus $case.q $iters $case.t 2>&1 | Out-String).Trim()
  $line = "block=$block order=$order $subject $corpusId $($case.q) t=$($case.t): $text"
  Write-Host $line
  Add-Content -LiteralPath $log -Value $line
  if ($text -notmatch 'us_per_op=') {
    Add-Content -LiteralPath $log -Value "FAIL parse $subject $corpusId"
    return
  }
  $m = @{}
  foreach ($pair in ($text -split '\s+')) {
    if ($pair -match '^(\w+)=(.+)$') { $m[$matches[1]] = $matches[2] }
  }
  $rec = [ordered]@{
    subject=$subject; block=$block; order=$order
    corpus=$corpusId; query=$case.q; threads=[int]$case.t; iterations=[int]$iters
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
  foreach ($c in $corpora) {
    if (-not (Test-Path $c.path)) {
      Add-Content -LiteralPath $log -Value "MISSING $($c.path)"
      continue
    }
    foreach ($case in $cases) {
      $iters = [int]$c[$case.key]
      if ($b % 2 -eq 1) {
        Run-One 'upstream' $up $c.id $c.path $case $b 1 $iters
        Run-One 'optimized' $opt $c.id $c.path $case $b 2 $iters
      } else {
        Run-One 'optimized' $opt $c.id $c.path $case $b 1 $iters
        Run-One 'upstream' $up $c.id $c.path $case $b 2 $iters
      }
    }
  }
}
Write-Host "WROTE $out"
Add-Content -LiteralPath $log -Value "WROTE $out"
