# MeeTARA Lab - Voice & Translation Testing PowerShell Script
# Run this script to test your voice and translation models

Write-Host "MeeTARA Lab - Voice & Translation Testing" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "models/production")) {
    Write-Host "ERROR: Please run this script from the meetara-lab root directory" -ForegroundColor Red
    exit 1
}

# Function to test GGUF model
function Test-GGUFModel {
    Write-Host "`nTesting GGUF Model..." -ForegroundColor Yellow
    
    $llamaPath = "llama.cpp/build/bin/llama-cli.exe"
    $ggufPath = "models/production/B_universal/meetara-qwen2.5-7B-instruct-Q4_K_M-20250809.gguf"
    
    # Check if llama.cpp is built
    if (-not (Test-Path $llamaPath)) {
        Write-Host "WARNING: llama-cli.exe not found. Please build llama.cpp first." -ForegroundColor Yellow
        Write-Host "   Run: cd llama.cpp && mkdir build && cd build && cmake .. && cmake --build . --config Release" -ForegroundColor Gray
        return $false
    }
    
    # Check if GGUF model exists
    if (-not (Test-Path $ggufPath)) {
        Write-Host "ERROR: GGUF model not found: $ggufPath" -ForegroundColor Red
        return $false
    }
    
    $modelSize = (Get-Item $ggufPath).Length / 1GB
    Write-Host "OK: Found GGUF model: $(Split-Path $ggufPath -Leaf)" -ForegroundColor Green
    Write-Host "OK: Model size: $([math]::Round($modelSize, 1)) GB" -ForegroundColor Green
    
    # Test simple inference
    $testPrompt = "Translate 'Hello, how are you?' to Hindi"
    Write-Host "Testing with prompt: $testPrompt" -ForegroundColor Cyan
    
    try {
        $cmd = @(
            $llamaPath,
            "-m", $ggufPath,
            "-p", $testPrompt,
            "-n", "50",   # Max tokens
            "-t", "4",    # Threads
            "--temp", "0.7"  # Temperature
        )
        
        Write-Host "Running: $($cmd -join ' ')" -ForegroundColor Gray
        
        $result = & $llamaPath "-m" $ggufPath "-p" $testPrompt "-n" "50" "-t" "4" "--temp" "0.7" 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "OK: GGUF inference successful!" -ForegroundColor Green
            Write-Host "Output: $($result -join "`n")" -ForegroundColor White
            return $true
        } else {
            Write-Host "ERROR: GGUF inference failed with exit code: $LASTEXITCODE" -ForegroundColor Red
            Write-Host "Error: $result" -ForegroundColor Red
            return $false
        }
        
    } catch {
        Write-Host "ERROR: Error running GGUF inference: $_" -ForegroundColor Red
        return $false
    }
}

# Function to test translation models
function Test-TranslationModels {
    Write-Host "`nTesting Translation Models..." -ForegroundColor Yellow
    
    $hiPath = "models/production/translation_models/translation_bundle_20250809_222911/hi_model"
    $tePath = "models/production/translation_models/translation_bundle_20250809_222911/te_model"
    
    # Check Hindi model
    if (Test-Path $hiPath) {
        Write-Host "OK: Hindi model found: $hiPath" -ForegroundColor Green
        $modelFile = Join-Path $hiPath "model.pt"
        if (Test-Path $modelFile) {
            $size = (Get-Item $modelFile).Length / 1MB
            Write-Host "   Model size: $([math]::Round($size, 1)) MB" -ForegroundColor Gray
        }
    } else {
        Write-Host "ERROR: Hindi model not found" -ForegroundColor Red
    }
    
    # Check Telugu model
    if (Test-Path $tePath) {
        Write-Host "OK: Telugu model found: $tePath" -ForegroundColor Green
        $modelFile = Join-Path $tePath "model.pt"
        if (Test-Path $modelFile) {
            $size = (Get-Item $modelFile).Length / 1MB
            Write-Host "   Model size: $([math]::Round($size, 1)) MB" -ForegroundColor Gray
        }
    } else {
        Write-Host "ERROR: Telugu model not found" -ForegroundColor Red
        return $false
    }
    
    # Check bundle config
    $bundleConfig = Join-Path (Split-Path $hiPath -Parent) "bundle_config.json"
    if (Test-Path $bundleConfig) {
        try {
            $config = Get-Content $bundleConfig | ConvertFrom-Json
            Write-Host "OK: Bundle config: $($config.bundle_name)" -ForegroundColor Green
            Write-Host "   Languages: $($config.languages.PSObject.Properties.Name -join ', ')" -ForegroundColor Gray
        } catch {
            Write-Host "WARNING: Error reading bundle config: $_" -ForegroundColor Yellow
        }
    }
    
    return $true
}

# Function to test speech models
function Test-SpeechModels {
    Write-Host "`nTesting Speech Models..." -ForegroundColor Yellow
    
    $speechPath = "models/production/speech_models"
    if (Test-Path $speechPath) {
        Write-Host "OK: Speech models directory found: $speechPath" -ForegroundColor Green
        
        # Check subdirectories
        @("emotion", "voice", "routing") | ForEach-Object {
            $subdirPath = Join-Path $speechPath $_
            if (Test-Path $subdirPath) {
                Write-Host "   OK: ${_}: $subdirPath" -ForegroundColor Green
            } else {
                Write-Host "   WARNING: ${_}: Not found" -ForegroundColor Yellow
            }
        }
        
        # Check speech config
        $speechConfig = Join-Path $speechPath "speech_config.json"
        if (Test-Path $speechConfig) {
            try {
                $config = Get-Content $speechConfig | ConvertFrom-Json
                $voiceCount = ($config.voice_mapping.PSObject.Properties | Measure-Object).Count
                Write-Host "OK: Speech config loaded: $voiceCount voice mappings" -ForegroundColor Green
            } catch {
                Write-Host "WARNING: Error reading speech config: $_" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "ERROR: Speech models directory not found" -ForegroundColor Red
    }
    
    return $true
}

# Function to check llama.cpp build
function Test-LlamaCppBuild {
    Write-Host "`nChecking llama.cpp build..." -ForegroundColor Yellow
    
    $llamaDir = "llama.cpp"
    if (-not (Test-Path $llamaDir)) {
        Write-Host "ERROR: llama.cpp directory not found" -ForegroundColor Red
        Write-Host "   Please clone: git clone https://github.com/ggerganov/llama.cpp.git" -ForegroundColor Gray
        return $false
    }
    
    $buildDir = Join-Path $llamaDir "build"
    if (-not (Test-Path $buildDir)) {
        Write-Host "WARNING: llama.cpp build directory not found" -ForegroundColor Yellow
        Write-Host "   Please build: cd llama.cpp && mkdir build && cd build && cmake .. && cmake --build . --config Release" -ForegroundColor Gray
        return $false
    }
    
    $llamaCli = Join-Path $buildDir "bin/llama-cli.exe"
    if (-not (Test-Path $llamaCli)) {
        Write-Host "WARNING: llama-cli.exe not found in build/bin" -ForegroundColor Yellow
        Write-Host "   Please build: cd llama.cpp/build && cmake .. && cmake --build . --config Release" -ForegroundColor Gray
        return $false
    }
    
    Write-Host "OK: llama.cpp is built and ready" -ForegroundColor Green
    Write-Host "   llama-cli.exe: $llamaCli" -ForegroundColor Gray
    return $true
}

# Function to run Python tests
function Test-PythonComponents {
    Write-Host "`nTesting Python Components..." -ForegroundColor Yellow
    
    # Test if Python is available
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "OK: Python available: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Python not available" -ForegroundColor Red
        return $false
    }
    
    # Test quick test script
    $quickTestScript = "tests/quick_test.py"
    if (Test-Path $quickTestScript) {
        Write-Host "Running Python quick test..." -ForegroundColor Cyan
        try {
            $result = python $quickTestScript 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "OK: Python quick test completed successfully" -ForegroundColor Green
                return $true
            } else {
                Write-Host "ERROR: Python quick test failed" -ForegroundColor Red
                Write-Host "Output: $result" -ForegroundColor Red
                return $false
            }
        } catch {
            Write-Host "ERROR: Error running Python test: $_" -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "WARNING: Quick test script not found: $quickTestScript" -ForegroundColor Yellow
        return $false
    }
}

# Main testing function
function Start-MeeTARATesting {
    Write-Host "Starting MeeTARA Lab Testing..." -ForegroundColor Green
    
    $results = @{
        "llama_cpp" = Test-LlamaCppBuild
        "gguf_model" = Test-GGUFModel
        "translation_models" = Test-TranslationModels
        "speech_models" = Test-SpeechModels
        "python_components" = Test-PythonComponents
    }
    
    Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
    Write-Host "TEST RESULTS SUMMARY" -ForegroundColor Green
    Write-Host ("=" * 60) -ForegroundColor Cyan
    
    foreach ($testName in $results.Keys) {
        $result = $results[$testName]
        $status = if ($result) { "PASS" } else { "FAIL" }
        $displayName = ($testName -replace "_", " ").ToUpper()
        Write-Host "$status $displayName" -ForegroundColor $(if ($result) { "Green" } else { "Red" })
    }
    
    Write-Host ("=" * 60) -ForegroundColor Cyan
    
    # Summary
    $passed = ($results.Values | Where-Object { $_ -eq $true }).Count
    $total = $results.Count
    Write-Host "Overall: $passed/$total tests passed" -ForegroundColor $(if ($passed -eq $total) { "Green" } else { "Yellow" })
    
    if ($passed -eq $total) {
        Write-Host "All tests passed! Your MeeTARA Lab is ready for voice and translation testing." -ForegroundColor Green
    } else {
        Write-Host "Some tests failed. Please check the issues above." -ForegroundColor Yellow
    }
    
    # Save results
    try {
        $resultsFile = "tests/powershell_test_results.json"
        $results | ConvertTo-Json | Out-File -FilePath $resultsFile -Encoding UTF8
        Write-Host "`nResults saved to: $resultsFile" -ForegroundColor Green
    } catch {
        Write-Host "`nFailed to save results: $_" -ForegroundColor Yellow
    }
    
    return $results
}

# Run the tests
try {
    $testResults = Start-MeeTARATesting
    exit 0
} catch {
    Write-Host "ERROR: Testing failed with error: $_" -ForegroundColor Red
    exit 1
}
