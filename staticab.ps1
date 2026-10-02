$k = 'C:\Users\kanal\Projects\fff-upstream\.fff-bench\profile-driver\target\release\rank.exe'
$b = 'C:\Users\kanal\Projects\fff-base\.fff-bench\profile-driver\target\release\rank.exe'
$u = 'C:\Users\kanal\Projects\fff-main-upstream\.fff-bench\profile-driver\target\release\rank.exe'
$sr = 'C:\Users\kanal\Projects\fff-upstream\.fff-bench\cache\static-repo'
$out = @()
foreach ($rep in 1..4) {
  foreach ($pair in @(@('base',$b), @('simd',$k), @('upstream',$u))) {
    $lines = & $pair[1] $sr 40 1 10 'AGENTS.md' 'readme' 'score' 'file_picker' 2>&1
    $q = ''
    foreach ($l in $lines) {
      if ($l -match '^timing query="([^"]+)"') { $q = $Matches[1] }
      if ($l -match 'p50_us=([0-9.]+)') { $out += ('{0},{1},{2}' -f $pair[0], $q, $Matches[1]) }
    }
  }
}
$out | Set-Content 'C:\Users\kanal\Projects\fff-upstream\static_ab.txt'