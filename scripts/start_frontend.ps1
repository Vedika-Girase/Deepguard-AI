$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\frontend"
if (!(Test-Path node_modules)) { npm install }
npm run dev
