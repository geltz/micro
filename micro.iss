[Setup]
AppId={{F08008C0-CB3E-4A3F-BC48-416D3376F367}
AppName=micro
AppVersion=1.3
AppPublisher=geltz
DefaultDirName={autopf}\micro
SetupIconFile=micro.ico
DefaultGroupName=micro
Compression=lzma2/ultra64
SolidCompression=yes
OutputDir=.
OutputBaseFilename=micro_setup_1.3
WizardStyle=modern
UninstallDisplayIcon={app}\micro.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\micro\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\micro"; Filename: "{app}\micro.exe"
Name: "{group}\{cm:UninstallProgram,micro}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\micro"; Filename: "{app}\micro.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\micro.exe"; Description: "{cm:LaunchProgram,micro}"; Flags: nowait postinstall skipifsilent