; Build on Windows after tools/package.py. No system Python or GPU SDK installs.
#ifndef Payload
  #define Payload "..\..\dist\windows-x64\MIA"
#endif
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId=io.github.jjeyxb.mia
AppName=MIA
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\MIA
DisableDirPage=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
AppMutex=io.github.jjeyxb.mia
UninstallDisplayIcon={app}\program\MIA.exe
OutputBaseFilename=MIA-{#AppVersion}-windows-x64-setup
OutputDir=..\..\dist\installers
Compression=lzma2
SolidCompression=yes
DiskSpanning=yes
DiskSliceSize=1900000000
CloseApplications=no

[Files]
Source: "{#Payload}\program\*"; DestDir: "{app}\program"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Payload}\install-info.json"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{userprograms}\MIA"; Filename: "{app}\program\MIA.exe"; Parameters: "--live"
Name: "{userdesktop}\MIA"; Filename: "{app}\program\MIA.exe"; Parameters: "--live"; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "建立桌面捷徑"; Flags: unchecked

[Run]
Filename: "{app}\program\MIA.exe"; Parameters: "--live"; Description: "啟動 MIA"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; These are exclusively MIA-owned subdirectories, never {app}\*.
Type: filesandordirs; Name: "{app}\program"
Type: filesandordirs; Name: "{app}\components"
Type: filesandordirs; Name: "{app}\updates"
Type: files; Name: "{app}\install-info.json"

[Code]
function NextButtonClick(CurPageID: Integer): Boolean;
var Root, Probe: String;
begin
  Result := True;
  if CurPageID <> wpSelectDir then exit;
  Root := RemoveBackslashUnlessRoot(WizardDirValue);
  if (Length(Root) <= 3) or (CompareText(Root, ExpandConstant('{win}')) = 0) or
     (CompareText(Root, ExpandConstant('{userprofile}')) = 0) then begin
    MsgBox('請選擇 MIA 專用資料夾。', mbError, MB_OK);
    Result := False;
    exit;
  end;
  if (DirExists(AddBackslash(Root) + 'program') or
      DirExists(AddBackslash(Root) + 'userdata') or
      DirExists(AddBackslash(Root) + 'components')) and
      not FileExists(AddBackslash(Root) + 'install-info.json') then begin
    MsgBox('這個資料夾已有其他內容，請選擇空白的 MIA 專用資料夾。', mbError, MB_OK);
    Result := False;
    exit;
  end;
  if not ForceDirectories(Root) then begin
    MsgBox('無法建立安裝資料夾，請選擇可寫入的位置。', mbError, MB_OK);
    Result := False;
    exit;
  end;
  Probe := AddBackslash(Root) + '.mia-write-test';
  Result := SaveStringToFile(Probe, 'MIA', False);
  DeleteFile(Probe);
  if not Result then MsgBox('此位置無法寫入設定與錄影，請另選資料夾。', mbError, MB_OK);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    if MsgBox('是否一併刪除 MIA 的設定、登入資料、錄影與紀錄？選擇「否」會保留。',
      mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      DelTree(ExpandConstant('{app}\userdata'), True, True, True);
end;
