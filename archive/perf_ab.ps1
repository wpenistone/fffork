# Deep perf regression check: interleaved A/B, per-query p50, several
# thread counts and corpora. Interleaving matters - this box drifts 20-30%
# run to run, and a blocked-then-A-then-B protocol hides that.
$repo = 'C:\Users\kanal\Projects\fff-upstream'
$up   = 'C:\Users\kanal\Projects\fff-main-upstream\.fff-bench\profile-driver\target\release\rank.exe'
$kn   = "$repo\.fff-bench\profile-driver\target\release\rank.exe"

$corpora = [ordered]@{
  'fff-1.8k'   = $repo
  'sumo-227k'  = 'C:\Users\kanal\Projects\sumo_fork'
  'squad-107k' = 'C:\Users\kanal\Projects\squad-ds-re'
  'opencode'   = 'C:\Users\kanal\src\opencode'
}
$queries = @('score','test','file_picker.rs','conf','index')
$iters = 6
$reps  = 3

function P50($exe,$corpus,$th,$q) {
  $vals = @()
  for ($r=0; $r -lt $reps; $r++) {
    $out = & $exe $corpus $iters $th 100 $q 2>$null
    foreach ($l in $out) {
      if ($l -match 'p50_us=([0-9.]+)') { $vals += [double]$Matches[1] }
    }
  }
  if ($vals.Count -eq 0) { return -1 }
  ($vals | Sort-Object)[[int]($vals.Count/2)]
}

$rows = @()
foreach ($c in $corpora.GetEnumerator()) {
  foreach ($th in 1,4) {
    foreach ($q in $queries) {
      # interleave: K,U,K,U,K,U and take the median of each side
      $k = P50 $kn $c.Value $th $q
      $u = P50 $up $c.Value $th $q
      $rows += [pscustomobject]@{ corpus=$c.Key; th=$th; q=$q; keep=$k; up=$u; ratio=[math]::Round($u/$k,2) }
      "{0,-11} {1}T {2,-16} keep={3,9:N0}us up={4,9:N0}us  {5,5}x" -f $c.Key,$th,$q,$k,$u,($u/$k)
    }
  }
}
""
$rows | Group-Object corpus,th | ForEach-Object {
  $g=$_.Group
  $worst = $g | Sort-Object ratio | Select-Object -First 1
  $best  = $g | Sort-Object ratio -Descending | Select-Object -First 1
  $gm = [math]::Round((($g | Measure-Object ratio -Average).Average),2)
  "{0,-18} mean={1,6}x  best={2,6}x ({3})  worst={4,6}x ({5})" -f $_.Name,$gm,$best.ratio,$best.q,$worst.ratio,$worst.q
}