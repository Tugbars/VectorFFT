param([string]$Pdf, [string]$OutPrefix, [int]$Width = 1240, [int]$First = 1, [int]$Last = 0)
# Rasterize pages of a PDF to PNG with the Windows.Data.Pdf WinRT API that ships with Windows
# (nothing to install): <OutPrefix>_p<N>.png for pages First..Last (Last 0 = the last page).
# usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/pdfpages.ps1 -Pdf <file.pdf> -OutPrefix <dir>/page
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$t) {
  $m = $asTaskGeneric.MakeGenericMethod($t)
  $task = $m.Invoke($null, @($op)); $task.Wait(-1) | Out-Null; $task.Result
}
$asTaskAction = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction' })[0]
function AwaitAction($op) { $task = $asTaskAction.Invoke($null, @($op)); $task.Wait(-1) | Out-Null }

$full = [System.IO.Path]::GetFullPath($Pdf)
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($full)) ([Windows.Storage.StorageFile])
$doc = Await ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($file)) ([Windows.Data.Pdf.PdfDocument])
if ($Last -le 0 -or $Last -gt $doc.PageCount) { $Last = $doc.PageCount }
Write-Output ("pages: " + $doc.PageCount)
for ($i = $First - 1; $i -lt $Last; $i++) {
  $page = $doc.GetPage($i)
  $opts = New-Object Windows.Data.Pdf.PdfPageRenderOptions
  $opts.DestinationWidth = [uint32]$Width
  $opts.BackgroundColor = [Windows.UI.Color]::FromArgb(255, 255, 255, 255)
  $stream = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
  AwaitAction ($page.RenderToStreamAsync($stream, $opts))
  $net = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($stream.GetInputStreamAt(0))
  $out = "{0}_p{1}.png" -f [System.IO.Path]::GetFullPath($OutPrefix), ($i + 1)
  $fs = [System.IO.File]::Create($out); $net.CopyTo($fs); $fs.Close(); $net.Close()
  Write-Output ("wrote " + $out)
  $page.Dispose()
}
