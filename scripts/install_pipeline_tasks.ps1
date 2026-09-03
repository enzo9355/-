[CmdletBinding(SupportsShouldProcess)]
param(
  [string]$DataRoot = 'D:\AbsorbData',
  [ValidateSet('Saturday','Sunday')][string]$WeeklyDay = 'Saturday'
)
$ErrorActionPreference = 'Stop'
if ($DataRoot -ne 'D:\AbsorbData') { throw 'Data root is not allowlisted' }

function Get-AbsorbPipelineTaskDefinitions {
  [CmdletBinding()]
  param([Parameter(Mandatory)][ValidateSet('Saturday','Sunday')][string]$WeeklyDay)

  $DefaultExecutionTimeLimit = New-TimeSpan -Hours 4
  return @(
    @{ Name='ABSORB-TW-PostClose'; Job='TW-PostClose'; Time='17:10'; RepetitionInterval='PT20M'; RepetitionDuration='PT4H50M'; ExecutionTimeLimit=$DefaultExecutionTimeLimit },
    @{ Name='ABSORB-TW-PreMarket'; Job='TW-PreMarket'; Time='07:30'; RepetitionInterval='PT10M'; RepetitionDuration='PT1H20M'; ExecutionTimeLimit=$DefaultExecutionTimeLimit },
    @{ Name='ABSORB-TW-ObservationRecovery'; Job='TW-ObservationRecovery'; Time='06:15'; RepetitionInterval='PT10M'; RepetitionDuration='PT15M'; ExecutionTimeLimit=$DefaultExecutionTimeLimit },
    @{ Name='ABSORB-FullBacktest'; Job='FullBacktest'; Time='22:30'; ExecutionTimeLimit=(New-TimeSpan -Minutes 225); Enabled=$false },
    @{ Name='ABSORB-US-Daily'; Job='US-Daily'; Time='05:30'; ExecutionTimeLimit=$DefaultExecutionTimeLimit },
    # Yahoo can finalize daily Close fields near New York midnight; cover both EDT and EST.
    @{ Name='ABSORB-US-PostClose'; Job='US-PostClose'; Time='09:00'; RepetitionInterval='PT20M'; RepetitionDuration='PT5H00M'; ExecutionTimeLimit=$DefaultExecutionTimeLimit },
    @{ Name='ABSORB-US-PreMarket'; Job='US-PreMarket'; Time='20:30'; ExecutionTimeLimit=$DefaultExecutionTimeLimit },
    @{ Name='ABSORB-WeeklyModel'; Job='WeeklyModel'; Time='18:00'; Days=$WeeklyDay; ExecutionTimeLimit=$DefaultExecutionTimeLimit },
    @{ Name='ABSORB-ReportUploadRecovery'; Job='ReportUploadRecovery'; Time='09:35'; ExecutionTimeLimit=$DefaultExecutionTimeLimit }
  )
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$Principal = New-ScheduledTaskPrincipal -UserId $Identity.Name -LogonType Interactive -RunLevel Limited
$TaskWrapper = Join-Path $PSScriptRoot 'invoke_pipeline_task.ps1'
if (-not (Test-Path -LiteralPath $TaskWrapper -PathType Leaf)) { throw "Task wrapper not found: $TaskWrapper" }
$HiddenLauncher = Join-Path $PSScriptRoot 'run_hidden.vbs'
if (-not (Test-Path -LiteralPath $HiddenLauncher -PathType Leaf)) { throw "Hidden launcher not found: $HiddenLauncher" }
$WscriptExe = (Get-Command wscript.exe -ErrorAction Stop).Source
$PowerShellExe = (Get-Command powershell.exe -ErrorAction Stop).Source
$Definitions = @(Get-AbsorbPipelineTaskDefinitions -WeeklyDay $WeeklyDay)
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
  } else {
    New-ScheduledTaskTrigger -Daily -At $At
  }
  $Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit $Definition.ExecutionTimeLimit -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10)
  $Settings.StartWhenAvailable = $true
  $Settings.WakeToRun = $true
  if ($PSCmdlet.ShouldProcess($Definition.Name, 'Register shadow pipeline task')) {
    Register-ScheduledTask -TaskName $Definition.Name -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force | Out-Null
    $TaskXml = [xml](schtasks /query /tn "\$($Definition.Name)" /xml)
    $Namespace = New-Object Xml.XmlNamespaceManager($TaskXml.NameTable)
    $Namespace.AddNamespace('t', 'http://schemas.microsoft.com/windows/2004/02/mit/task')
    $Changed = $false
    $TriggerNode = $TaskXml.SelectSingleNode('//t:CalendarTrigger', $Namespace)
    if ($TriggerNode -and $Definition.RepetitionInterval -and $Definition.RepetitionDuration -and -not $TaskXml.SelectSingleNode('//t:Repetition', $Namespace)) {
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
      $Changed = $true
    }
    $SettingsNode = $TaskXml.SelectSingleNode('//t:Settings', $Namespace)
    if ($null -eq $SettingsNode) { throw 'Scheduled task settings node is missing' }
    foreach ($BatteryNodeName in @('DisallowStartIfOnBatteries', 'StopIfGoingOnBatteries')) {
      $BatteryNode = $SettingsNode.SelectSingleNode("t:$BatteryNodeName", $Namespace)
      if ($null -eq $BatteryNode) {
        $BatteryNode = $TaskXml.CreateElement($BatteryNodeName, 'http://schemas.microsoft.com/windows/2004/02/mit/task')
        $ExecutionTimeLimitNode = $SettingsNode.SelectSingleNode('t:ExecutionTimeLimit', $Namespace)
        if ($null -eq $ExecutionTimeLimitNode) { throw 'Scheduled task execution limit node is missing' }
        $SettingsNode.InsertBefore($BatteryNode, $ExecutionTimeLimitNode) | Out-Null
        $Changed = $true
      }
      if ([string]$BatteryNode.InnerText -ne 'false') {
        $BatteryNode.InnerText = 'false'
        $Changed = $true
      }
    }
    if ($Changed) {
      Register-ScheduledTask -TaskName $Definition.Name -Xml $TaskXml.OuterXml -Force | Out-Null
    }
    if ($Definition.ContainsKey('Enabled') -and $Definition.Enabled -eq $false) {
      try {
        Disable-ScheduledTask -TaskName $Definition.Name -ErrorAction Stop | Out-Null
      }
      catch {
        throw "Unable to disable completed scheduled task $($Definition.Name)"
      }
      try {
        [xml]$RegistrationXml = schtasks /query /tn "\$($Definition.Name)" /xml
        $EnabledNode = $RegistrationXml.SelectSingleNode(
          "/*[local-name()='Task']/*[local-name()='Settings']/*[local-name()='Enabled']"
        )
      }
      catch {
        throw "Unable to verify completed scheduled task registration $($Definition.Name)"
      }
      if (
        $null -eq $EnabledNode -or
        ([string]$EnabledNode.InnerText).Trim().ToLowerInvariant() -ne 'false'
      ) {
        throw "Scheduled task $($Definition.Name) must remain disabled"
      }
    }
  }
}
