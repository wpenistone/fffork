# G-01 corpus assembly, phase 1: copy the six real trees.
# The generated Chromium-shaped fill is added by gen_gold_fill.py after the
# copied base is measured through the release driver.
$ErrorActionPreference = 'Stop'

$root = 'D:\fff-corpus\gold-500k'
$build = 'C:\Users\kanal\Projects\fff-upstream\scratch\corpora\gold-500k-build'
New-Item -ItemType Directory -Force -Path $root, $build | Out-Null

$sources = @(
    @{ Name = 'sumo';                 Path = 'C:\Users\kanal\Projects\sumo_fork';                                          Indexed = 227405; Raw = 227440;  MB = 2099.2 },
    @{ Name = 'python-site-packages'; Path = 'C:\Users\kanal\AppData\Local\Programs\Python\Python313\Lib\site-packages'; Indexed = 45204;  Raw = 71881;   MB = 4198.4 },
    @{ Name = 'web-vendor';           Path = 'C:\Users\kanal\node_modules';                                                Indexed = 39082;  Raw = 56550;   MB = 4720.6 },
    @{ Name = 'vscode-extensions';    Path = 'C:\Users\kanal\.vscode\extensions';                                          Indexed = 18513;  Raw = 31503;   MB = 2334.7 },
    @{ Name = 'squad-ds-re';          Path = 'C:\Users\kanal\Projects\squad-ds-re';                                        Indexed = 107633; Raw = 107751;  MB = 1904.6 },
    @{ Name = '911-pager-v2';         Path = 'C:\Users\kanal\Projects\911-pager-v2';                                       Indexed = 51430;  Raw = 51431;   MB = 36.5 }
)

$rows = @()
foreach ($s in $sources) {
    if (-not (Test-Path -LiteralPath $s.Path)) { throw "missing source: $($s.Path)" }
    $dest = Join-Path $root $s.Name
    $log = Join-Path $build ("copy-" + $s.Name + ".log")
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $rcArgs = @(
        $s.Path, $dest,
        '/E', '/COPY:DAT', '/DCOPY:DAT', '/XJ',
        '/R:1', '/W:1', '/MT:16',
        '/NFL', '/NDL', '/NJH', '/NJS', '/NP',
        '/XD', '.git',
        ('/LOG:' + $log)
    )
    & robocopy @rcArgs | Out-Null
    $code = $LASTEXITCODE
    $sw.Stop()
    if ($code -ge 8) { throw "robocopy failed for $($s.Name): exit $code (see $log)" }
    $rows += [pscustomobject]@{
        name = $s.Name
        source = $s.Path
        destination = $dest
        source_indexed = $s.Indexed
        source_raw = $s.Raw
        source_mb = $s.MB
        robocopy_exit = $code
        seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    }
    Write-Host ("copied {0}: exit {1}, {2}s" -f $s.Name, $code, [math]::Round($sw.Elapsed.TotalSeconds, 1))
}

$rows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $build 'copy-summary.json') -Encoding UTF8
Write-Host "copy phase complete: $root"
