[CmdletBinding(SupportsShouldProcess)]
param(
  [string]$DataRoot = 'D:\AbsorbData',
  [ValidateSet('Saturday','Sunday')][string]$WeeklyDay = 'Saturday'
)
$ErrorActionPreference = 'Stop'
if ($DataRoot -ne 'D:\AbsorbData') { throw 'Data root is not allowlisted' }
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$Principal = New-ScheduledTaskPrincipal -UserId $Identity.Name -LogonType Interactive -RunLevel Limited
$TaskWrapper = Join-Path $PSScriptRoot 'invoke_pipeline_task.ps1'
if (-not (Test-Path -LiteralPath $TaskWrapper -PathType Leaf)) { throw "Task wrapper not found: $TaskWrapper" }
$HiddenLauncher = Join-Path $PSScriptRoot 'run_hidden.vbs'
if (-not (Test-Path -LiteralPath $HiddenLauncher -PathType Leaf)) { throw "Hidden launcher not found: $HiddenLauncher" }
$WscriptExe = (Get-Command wscript.exe -ErrorAction Stop).Source
$PowerShellExe = (Get-Command powershell.exe -ErrorAction Stop).Source
$Definitions = @(
  @{ Name='ABSORB-TW-PostClose'; Job='TW-PostClose'; Time='17:10'; RepetitionInterval='PT20M'; RepetitionDuration='PT4H50M' },
  @{ Name='ABSORB-TW-PreMarket'; Job='TW-PreMarket'; Time='07:30' },
  @{ Name='ABSORB-FullBacktest'; Job='FullBacktest'; Time='22:30'; RepeatMinutes=1 },
  @{ Name='ABSORB-US-Daily'; Job='US-Daily'; Time='05:30' },
  @{ Name='ABSORB-US-PostClose'; Job='US-PostClose'; Time='05:00'; RepetitionInterval='PT20M'; RepetitionDuration='PT4H00M' },
  @{ Name='ABSORB-US-PreMarket'; Job='US-PreMarket'; Time='20:30' },
  @{ Name='ABSORB-WeeklyModel'; Job='WeeklyModel'; Time='18:00'; Days=$WeeklyDay },
  @{ Name='ABSORB-ReportUploadRecovery'; Job='ReportUploadRecovery'; Time='09:35' }
)
foreach ($Definition in $Definitions) {
  $ActionArguments = @(
    '//B',
    '//NoLogo',
    "`"$HiddenLauncher`"",
    "`"$PowerShellExe`"",
    '-NoProfile',
    '-NonInteractive',
    '-WindowStyle',
    'Hidden',
    '-ExecutionPolicy',
    'Bypass',
    '-File',
    "`"$TaskWrapper`"",
    '-Job',
    "`"$($Definition.Job)`"",
    '-DataRoot',
    "`"$DataRoot`""
  ) -join ' '
  $Action = New-ScheduledTaskAction -Execute $WscriptExe -Argument $ActionArguments -WorkingDirectory $RepoRoot
  $At = [datetime]::ParseExact($Definition.Time, 'HH:mm', $null)
  $Trigger = if ($Definition.Days) {
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Definition.Days -At $At
  } elseif ($Definition.RepeatMinutes) {
    New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes $Definition.RepeatMinutes)
  } else {
    New-ScheduledTaskTrigger -Daily -At $At
  }
  $Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 4) -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10)
  $Settings.StartWhenAvailable = $true
  $Settings.WakeToRun = $true
  if ($PSCmdlet.ShouldProcess($Definition.Name, 'Register shadow pipeline task')) {
    Register-ScheduledTask -TaskName $Definition.Name -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force | Out-Null
    if ($Definition.RepetitionInterval -and $Definition.RepetitionDuration) {
      $TaskXml = [xml](schtasks /query /tn "\$($Definition.Name)" /xml)
      $Namespace = New-Object Xml.XmlNamespaceManager($TaskXml.NameTable)
      $Namespace.AddNamespace('t', 'http://schemas.microsoft.com/windows/2004/02/mit/task')
      $TriggerNode = $TaskXml.SelectSingleNode('//t:CalendarTrigger', $Namespace)
      if ($TriggerNode -and -not $TaskXml.SelectSingleNode('//t:Repetition', $Namespace)) {
        $RepetitionNode = $TaskXml.CreateElement('Repetition', 'http://schemas.microsoft.com/windows/2004/02/mit/task')
        $IntervalNode = $TaskXml.CreateElement('Interval', 'http://schemas.microsoft.com/windows/2004/02/mit/task')
        $IntervalNode.InnerText = $Definition.RepetitionInterval
        $DurationNode = $TaskXml.CreateElement('Duration', 'http://schemas.microsoft.com/windows/2004/02/mit/task')
        $DurationNode.InnerText = $Definition.RepetitionDuration
        $StopNode = $TaskXml.CreateElement('StopAtDurationEnd', 'http://schemas.microsoft.com/windows/2004/02/mit/task')
        $StopNode.InnerText = 'true'
        $RepetitionNode.AppendChild($IntervalNode) | Out-Null
        $RepetitionNode.AppendChild($DurationNode) | Out-Null
        $RepetitionNode.AppendChild($StopNode) | Out-Null
        $TriggerNode.AppendChild($RepetitionNode) | Out-Null
        Register-ScheduledTask -TaskName $Definition.Name -Xml $TaskXml.OuterXml -Force | Out-Null
      }
    }
  }
}
