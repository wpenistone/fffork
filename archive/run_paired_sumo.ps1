# July-style paired process runs: upstream vs keepers on frozen sumo 227k
$up = 'C:/Users/kanal/Projects/fff-upstream/.fff-bench/build/upstream/release/fff-profile-driver.exe'
$opt = 'C:/Users/kanal/Projects/fff-upstream/archive/bench-binaries/fff-profile-driver-keepers.exe'
$corpus = 'C:/Users/kanal/Projects/fff-upstream/.fff-bench/cache/sumo-worktree'
$out = 'C:/Users/kanal/Projects/fff-upstream/archive/paired-sumo-process-runs.jsonl'
Remove-Item -LiteralPath $out -ErrorAction SilentlyContinue

$cases = @(
  @{q='file_picker_'; n=500; t=1},
  @{q='file_picker_'; n=500; t=4},
  @{q='score'; n=200; t=1},
  @{q='score'; n=200; t=4}
)

function Run-One($subject, $exe, $case, $block, $order) {
  $text = (& $exe $corpus $case.q $case.n $case.t 2>&1 | Out-String)
  if ($text -notmatch 'us_per_op=([0-9.]+).*checksum=(\d+)') {
    Write-Host "FAIL $subject $($case.q) t=$($case.t): $text"
    return
  }
  $rec = [pscustomobject]@{
    block=$block; order=$order; subject=$subject
    query=$case.q; threads=$case.t; iters=$case.n
    us_per_op=[double]$matches[1]; checksum=[int64]$matches[2]
    ts=(Get-Date).ToString('o')
  }
  $rec | ConvertTo-Json -Compress | Add-Content -LiteralPath $out
  Write-Host ("block={0} order={1} {2,-8} q={3} t={4} us={5:N1} sum={6}" -f $block,$order,$subject,$case.q,$case.t,$rec.us_per_op,$rec.checksum)
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
