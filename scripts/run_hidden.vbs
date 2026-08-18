Option Explicit

Dim arguments, command, index, shell, exitCode
Set arguments = WScript.Arguments

If arguments.Count < 1 Then
    WScript.Quit 87
End If

command = QuoteArgument(arguments.Item(0))
For index = 1 To arguments.Count - 1
    command = command & " " & QuoteArgument(arguments.Item(index))
Next

Set shell = CreateObject("WScript.Shell")
exitCode = shell.Run(command, 0, True)
WScript.Quit exitCode

Function QuoteArgument(ByVal value)
    If InStr(value, Chr(34)) > 0 Then
        WScript.Quit 87
    End If
    QuoteArgument = Chr(34) & value & Chr(34)
End Function
