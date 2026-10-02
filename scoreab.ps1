$k = 'C:\Users\kanal\Projects\fff-upstream\.fff-bench\profile-driver\target\release\rank.exe'
$b = 'C:\Users\kanal\Projects\fff-base\.fff-bench\profile-driver\target\release\rank.exe'
$sr = 'C:\Users\kanal\Projects\fff-upstream\.fff-bench\cache\static-repo'
$mega = 'C:\Users\kanal\Projects\fff-upstream\.fff-bench\cache\mega-repo'
$out = @()
foreach ($set in @(@('small477', $sr, 'AGENTS.md','readme','score','file_picker','conf','lua_types'), @('mega170k', $mega, 'score','test','file_picker.rs','conf'))) {
  foreach ($rep in 1..3) {
    foreach ($pair in @(@('base',$b), @('new',$k))) {
      $lines = & $pair[1] $set[1] 20 1 10 $set[2..($set.Count-1)] 2>&1
      $q = ''
      foreach ($l in $lines) {
        if ($l -match '^timing query="([^"]+)"') { $q = $Matches[1] }
        if ($l -match 'p50_us=([0-9.]+)') { $out += ('{0},{1},{2},{3}' -f $set[0], $pair[0], $q, $Matches[1]) }
      }
    }
  }
}
$out | Set-Content 'C:\Users\kanal\Projects\fff-upstream\score_ab.txt'