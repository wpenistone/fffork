$k = 'C:\Users\kanal\Projects\fff-upstream\.fff-bench\profile-driver\target\release\rank.exe'
$u = 'C:\Users\kanal\Projects\fff-main-upstream\.fff-bench\profile-driver\target\release\rank.exe'
$corpora = @(
  @{id='sumo_fork-227k'; path='C:\Users\kanal\Projects\sumo_fork'},
  @{id='opencode-145k';  path='C:\Users\kanal\src\opencode'},
  @{id='kernel-100k';    path='C:\Users\kanal\Documents\antigravity\keen-oppenheimer'},
  @{id='squad-ds-107k';  path='C:\Users\kanal\Projects\squad-ds-re'}
)
$qs = @('score','mod','file_picker','controller')
$out = @()
foreach ($c in $corpora) {
  foreach ($th in @(1,4)) {
    foreach ($pair in @(@('keepers',$k), @('upstream',$u))) {
      $lines = & $pair[1] $c.path 10 $th 10 $qs 2>&1
      $q = ''
      foreach ($l in $lines) {
        if ($l -match '^timing query="([^"]+)"') { $q = $Matches[1] }
        if ($l -match 'p50_us=([0-9.]+)') { $out += ('{0},{1},{2},{3},{4}' -f $c.id, $th, $pair[0], $q, $Matches[1]) }
      }
    }
  }
  "done $($c.id)" | Out-File -Append 'C:\Users\kanal\Projects\fff-upstream\real_progress.txt'
}
$out | Set-Content 'C:\Users\kanal\Projects\fff-upstream\real_ab.txt'