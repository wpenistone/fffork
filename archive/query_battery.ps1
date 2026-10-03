# Realistic query battery: what someone actually types when they know
# roughly what they want but not the exact name.
# Each entry: query -> substring the right answer must contain.
$repo = 'C:\Users\kanal\Projects\fff-upstream'
$up   = 'C:\Users\kanal\Projects\fff-main-upstream\.fff-bench\profile-driver\target\release\rank.exe'
$kn   = "$repo\.fff-bench\profile-driver\target\release\rank.exe"

$cases = @(
  @('score','fff-core/src/score.rs'),
  @('file_picker','fff-core/src/file_picker.rs'),
  @('picker_ui','lua/fff/picker_ui'),
  @('conf','lua/fff/conf.lua'),
  @('frecncy','dbs/frecency.rs'),
  @('frecency','dbs/frecency.rs'),
  @('lua_types','crates/fff-nvim/src/lua_types.rs'),
  @('Makefile','Makefile'),
  @('git','fff-core/src/git.rs'),
  @('gate','fff-core/src/gate.rs'),
  @('machine','fff-core/src/machine.rs'),
  @('intent','fff-core/src/intent.rs'),
  @('grep','fff-core/src/grep'),
  @('AGENTS','AGENTS.md'),
  @('simd_path','simd_path.rs'),
  @('interseq','interseq_sw.rs'),
  @('presence','presence.rs'),
  @('watcher','watcher'),
  @('scroe','score.rs'),
  @('pickr','picker'),
  @('src/dbs','dbs/'),
  @('lua/fff','lua/fff'),
  @('types','types.rs'),
  @('parallel','parallelism.rs'),
  @('main','lua/fff/main.lua'),
  @('fuzzy','fuzzy'),
  @('bench','bench'),
  @('index','index'),
  @('watch','watch'),
  @('prefilter','prefilter')
)

function Get-Ranks($exe, $q, $limit) {
  $out = & $exe $repo 1 1 $limit $q 2>$null
  $paths = @()
  foreach ($l in $out) {
    if ($l -match '^\d+\t\d+\t(.+?)\t') { $paths += $Matches[1] }
  }
  return $paths
}

$rows = @()
foreach ($c in $cases) {
  $q = $c[0]; $want = $c[1]
  $pk = Get-Ranks $kn $q 20
  $pu = Get-Ranks $up $q 20
  $rk = -1; $ru = -1
  for ($i=0; $i -lt $pk.Count; $i++) { if ($pk[$i] -like "*$want*") { $rk = $i; break } }
  for ($i=0; $i -lt $pu.Count; $i++) { if ($pu[$i] -like "*$want*") { $ru = $i; break } }
  $rows += [pscustomobject]@{ query=$q; want=$want; keep=$rk; up=$ru; top_keep=$pk[0]; top_up=$pu[0] }
}
$rows | Format-Table -AutoSize | Out-String -Width 200

$kh = ($rows | Where-Object { $_.keep -eq 0 }).Count
$uh = ($rows | Where-Object { $_.up   -eq 0 }).Count
$kf = ($rows | Where-Object { $_.keep -lt 0 }).Count
$uf = ($rows | Where-Object { $_.up   -lt 0 }).Count
"hit@1  keepers=$kh/$($rows.Count)   upstream=$uh/$($rows.Count)"
"miss   keepers=$kf   upstream=$uf"