# Typing simulation: type a target filename one keystroke at a time and
# check the intended file is still visible. This is what the user actually
# experiences - they do not know the exact name, they type until it appears.
$repo = 'C:\Users\kanal\Projects\fff-upstream'
$up   = 'C:\Users\kanal\Projects\fff-main-upstream\.fff-bench\profile-driver\target\release\rank.exe'
$kn   = "$repo\.fff-bench\profile-driver\target\release\rank.exe"

# word -> what the user is trying to reach
$targets = [ordered]@{
  'file_picker.rs' = 'fff-core/src/file_picker.rs'
  'score.rs'       = 'fff-core/src/score.rs'
  'conf.lua'       = 'lua/fff/conf.lua'
  'gate.rs'        = 'fff-core/src/gate.rs'
  'frecency.rs'    = 'dbs/frecency.rs'
  'picker_ui'      = 'lua/fff/picker_ui'
}

function Get-Top($exe, $q, $limit) {
  $out = & $exe $repo 1 1 $limit $q 2>$null
  $paths = @()
  foreach ($l in $out) { if ($l -match '^\d+\t\d+\t(.+?)\t') { $paths += $Matches[1] } }
  return $paths
}

$total = 0; $kHit1 = 0; $uHit1 = 0; $kAny = 0; $uAny = 0
$firstK = @(); $firstU = @()
foreach ($t in $targets.GetEnumerator()) {
  $word = $t.Key; $want = $t.Value
  for ($i = 3; $i -le $word.Length; $i++) {
    $pre = $word.Substring(0, $i)
    $pk = Get-Top $kn $pre 10
    $pu = Get-Top $up $pre 10
    $total++
    $rk = ($pk | Select-Object -First 1) -like "*$want*"
    $ru = ($pu | Select-Object -First 1) -like "*$want*"
    $ak = ($pk -like "*$want*").Count -gt 0
    $au = ($pu -like "*$want*").Count -gt 0
    if ($rk) { $kHit1++ } ; if ($ru) { $uHit1++ }
    if ($ak) { $kAny++ } ; if ($au) { $uAny++ }
    # record the first prefix where each side puts it at #1
    if (-not $firstK.ContainsKey($word) -and $rk) { $firstK[$word] = $pre }
    if (-not $firstU.ContainsKey($word) -and $ru) { $firstU[$word] = $pre }
    if (-not $ak -or -not $au) {
      "{0,-16} {1,-16} keep1={2,-5} up1={3,-5} keepTop={4}" -f $word, $pre, $rk, $ru, ($pk | Select-Object -First 1)
    }
  }
}
""
"keystrokes tested: $total"
"hit@1  keepers=$kHit1 ($([math]::Round(100*$kHit1/$total))%)   upstream=$uHit1 ($([math]::Round(100*$uHit1/$total))%)"
"in top10 keepers=$kAny ($([math]::Round(100*$kAny/$total))%)   upstream=$uAny ($([math]::Round(100*$uAny/$total))%)"
""
"first prefix reaching #1:"
foreach ($t in $targets.Keys) {
  $k = if ($firstK.ContainsKey($t)) { $firstK[$t] } else { 'NEVER' }
  $u = if ($firstU.ContainsKey($t)) { $firstU[$t] } else { 'NEVER' }
  "{0,-16} keepers={1,-10} upstream={2}" -f $t, $k, $u
}