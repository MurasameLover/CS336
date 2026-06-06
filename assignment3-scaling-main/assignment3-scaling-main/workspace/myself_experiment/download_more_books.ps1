# Download many Gutenberg books in parallel and combine into a large corpus
$outputFile = "D:\CS336\assignment3-scaling-main\assignment3-scaling-main\workspace\myself_experiment\data_cache\big_text.txt"
$existing = Get-Content $outputFile -Raw -ErrorAction SilentlyContinue
$existingLen = if ($existing) { $existing.Length } else { 0 }
Write-Output "Existing corpus: $existingLen chars"

# IDs of books known to work with Project Gutenberg
# Shakespeare + classic English literature + philosophy + history
$bookIds = @(
    # We already have these, but adding more:
    1342, 11, 1661, 2701, 1400, 25344, 215, 73, 1232, 2852
    37106, 45839, 345, 174, 2678, 16713, 550, 4217
    # Additional books
    98, 120, 244, 768, 1260, 1459, 1581, 1740, 1837, 205
    219, 2554, 2641, 29765, 30254, 3207, 33283, 36, 3825, 3950
    408, 4085, 4218, 42324, 4280, 4300, 432, 43463, 46, 46852
    47135, 49106, 492, 50050, 507, 51060, 51785, 5200, 5230, 526
    53507, 54068, 546, 561, 56240, 5740, 57592, 57989, 58037, 58229
    58252, 58258, 58280, 58379, 58463, 58526, 58569, 58615, 5862, 58633
    58668, 58674, 58721, 58737, 594, 61179, 61877, 6370, 64317, 65048
    65233, 65328, 65359, 65392, 65441, 65611, 65737, 66, 66023, 66383
    66645, 66799, 66922, 66924, 67098, 67177, 67206, 67258, 67328, 67733
    67814, 67949, 68049, 68165, 68346, 68419, 68687, 68900, 69041, 69173
    69402, 69455, 69571, 69740, 69819, 69904, 69931, 69954, 69973, 70011
    70017, 70020, 70024, 70042, 70048, 70054, 70056, 70063, 70064, 70073
    70081, 70092, 70099, 70107, 70108, 70114, 70119, 70122, 70127, 70129
    70130, 70138, 70144, 70147, 70151, 70152, 70156, 70161, 70162, 70168
    70170, 70177, 70182, 70185, 70187, 70190, 70192, 70206, 70209, 70220
    70224, 70235, 70250, 70260, 70264, 70265, 70266, 70272, 70284, 70286
    70301, 70302, 70308, 70311, 70319, 70327, 70357, 70362, 70369, 70373
    70376, 70379, 70382, 70388, 70389, 70392, 70394, 70397, 70400, 70406
    70414, 70420, 70428, 70435, 70440, 70448, 70451, 70456, 70457, 70465
    70468, 70475, 70478, 70479, 70486, 70509, 70512, 70518, 70522, 70527
    70532, 70541, 70542, 70546, 70552, 70555, 70560, 70566, 70572, 70573
    70578, 70580, 70582, 70584, 70587, 70589, 70591, 70596, 70597, 70608
    70618, 70623, 70625, 70628, 70630, 70631, 70635, 70636, 70638, 70642
)

$wc = New-Object System.Net.WebClient
$wc.Headers.Add("User-Agent", "Mozilla/5.0")
$total = $existingLen
$ok = 0; $fail = 0

foreach ($id in $bookIds) {
    $url = "https://www.gutenberg.org/cache/epub/$id/pg$id.txt"
    try {
        $text = $wc.DownloadString($url)
        if ($existing) { $existing += "`n`n" + $text }
        else { $existing = $text }
        $total += $text.Length; $ok++
        Write-Output "OK #$id ($($text.Length) chars) - total: $total"
    } catch {
        $fail++
    }
    # Small delay to be polite
    Start-Sleep -Milliseconds 200
}

$existing | Set-Content -Path $outputFile -Encoding UTF8 | Out-Null
Write-Output "Done: $ok OK, $fail failed, $total total chars"
