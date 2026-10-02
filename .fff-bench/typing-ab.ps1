# Typing-latency A/B: the metric users actually feel.
#
# The per-query benchmark runs unrelated queries repeatedly, so every call is a
# full index scan and the live-narrow path never fires. Real use is a prefix
# sequence - "f", "fi", "fil", ... - where each keystroke extends the previous
# query and the live state carries over. This harness runs each keystroke
# exactly once, in order, so that state behaves as it does in nvim.
#
# Reports per-keystroke latency, the peak (the worst moment while typing) and
# the total for the whole word - the number that decides whether picking feels
# instant or laggy.
param(
  [string]$Keepers = 'C:\Users\kanal\Projects\fff-upstream\.fff-bench\profile-driver\target\release\rank.exe',
  [string]$Upstream = 'C:\Users\kanal\Projects\fff-main-upstream\.fff-bench\profile-driver\target\release\rank.exe',
  [int]$Threads = 4,
  [switch]$SkipUpstream
)

$corpora = @(
  @{ id = 'sumo_fork-227k'; path = 'C:\Users\kanal\Projects\sumo_fork'; word = 'file_picker' },
  @{ id = 'kernel-100k';    path = 'C:\Users\kanal\Documents\antigravity\keen-oppenheimer'; word = 'score' },
  @{ id = 'squad-ds-107k';  path = 'C:\Users\kanal\Projects\squad-ds-re'; word = 'test' },
  @{ id = 'opencode-6.6k';  path = 'C:\Users\kanal\src\opencode'; word = 'file' }
)

function Get-Prefixes([string]$word) {
  $out = @()
  for ($i = 1; $i -le $word.Length; $i++) { $out += $word.Substring(0, $i) }
  return $out
}

# Type the word once from scratch. Returns per-keystroke microseconds.
function Measure-Typing([string]$exe, [string]$corpus, [string]$word, [int]$th) {
  $qs = Get-Prefixes $word
  $lines = & $exe $corpus 1 $th 10 $qs 2>&1
  $per = @()
  $q = ''
  foreach ($l in $lines) {
    if ($l -match '^timing query="([^"]+)"') { $q = $Matches[1] }
    if ($l -match 'p50_us=([0-9.]+)') { $per += [pscustomobject]@{ q = $q; us = [double]$Matches[1] } }
  }
  return $per
}

$rows = @()
foreach ($c in $corpora) {
  if (-not (Test-Path $c.path)) { continue }
  # 3 passes, keep the best total: least machine noise.
  $bestK = $null; $bestKtot = [double]::MaxValue
  for ($p = 0; $p -lt 3; $p++) {
    $r = Measure-Typing $Keepers $c.path $c.word $Threads
    $tot = ($r | Measure-Object us -Sum).Sum
    if ($tot -lt $bestKtot) { $bestKtot = $tot; $bestK = $r }
  }
  $bestU = $null; $bestUtot = [double]::MaxValue
  if (-not $SkipUpstream -and (Test-Path $Upstream)) {
    for ($p = 0; $p -lt 3; $p++) {
      $r = Measure-Typing $Upstream $c.path $c.word $Threads
      $tot = ($r | Measure-Object us -Sum).Sum
      if ($tot -lt $bestUtot) { $bestUtot = $tot; $bestU = $r }
    }
  }
  $rows += [pscustomobject]@{
    corpus     = $c.id
    word       = $c.word
    keys       = $bestK.Count
    k_total_ms = [math]::Round($bestKtot / 1000, 1)
    k_peak_us  = [math]::Round(($bestK | Measure-Object us -Maximum).Maximum)
    u_total_ms = if ($bestU) { [math]::Round($bestUtot / 1000, 1) } else { $null }
    u_peak_us  = if ($bestU) { [math]::Round((($bestU | Measure-Object us -Maximum).Maximum)) } else { $null }
    total_x    = if ($bestU) { [math]::Round($bestUtot / $bestKtot, 2) } else { $null }
    k_sequence = (($bestK | ForEach-Object { [math]::Round($_.us) }) -join ' ')
  }
}

$rows | Format-Table -AutoSize | Out-String -Width 220
$rows | Export-Csv -NoTypeInformation -Path (Join-Path $PSScriptRoot 'typing-ab.csv')