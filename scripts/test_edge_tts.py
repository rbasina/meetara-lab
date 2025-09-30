#!/usr/bin/env python3
"""
Test Edge TTS Voices - MeeTARA Lab
Quick test script to verify Edge TTS voices work
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("[ERROR] edge-tts not installed. Install with: pip install edge-tts")
    sys.exit(1)

# MeeTARA Lab configured voices (updated with available voices)
MEETARA_VOICES = {
    "healthcare": "en-US-JennyNeural",
    "daily_life": "en-US-AriaNeural",
    "business": "en-US-GuyNeural",
    "education": "en-US-EmmaNeural",  # Fixed: MonicaNeural not available
    "creative": "en-US-MichelleNeural",  # Fixed: SaraNeural not available
    "technology": "en-US-ChristopherNeural",  # Fixed: JasonNeural not available
    "psychology": "en-US-AriaNeural",
    "sports": "en-US-EricNeural",  # Fixed: TonyNeural not available
    "legal": "en-US-BrianNeural",
    "travel": "en-US-AriaNeural"
}

async def test_voice(voice_name: str, text: str = "Hello, this is a test of the voice system."):
    """Test a specific Edge TTS voice"""
    print(f"\n[Testing] {voice_name}")
    print(f"Text: \"{text}\"")
    
    try:
        communicate = edge_tts.Communicate(text, voice_name)
        audio_data = b""
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        print(f"[OK] Generated {len(audio_data):,} bytes of audio")
        return True
    
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        return False

async def test_all_meetara_voices():
    """Test all MeeTARA Lab configured voices"""
    print("=" * 70)
    print("MEETARA LAB - EDGE TTS VOICE TEST")
    print("=" * 70)
    
    results = {}
    for category, voice in MEETARA_VOICES.items():
        text = f"Hello, I'm the {category} voice for MeeTARA Lab."
        success = await test_voice(voice, text)
        results[voice] = success
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for voice, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {voice}")
    
    print("=" * 70)
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print("=" * 70)
    
    return passed == total

async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Edge TTS voices")
    parser.add_argument("--voice", help="Specific voice to test (e.g., en-US-AriaNeural)")
    parser.add_argument("--text", default="Hello, testing voice", help="Text to synthesize")
    parser.add_argument("--all", action="store_true", help="Test all MeeTARA voices")
    parser.add_argument("--list", action="store_true", help="List configured voices")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nMeeTARA Lab Configured Voices:")
        print("-" * 50)
        for category, voice in MEETARA_VOICES.items():
            print(f"  {category:15s} -> {voice}")
        print("-" * 50)
        return
    
    if args.all:
        success = await test_all_meetara_voices()
        sys.exit(0 if success else 1)
    
    if args.voice:
        success = await test_voice(args.voice, args.text)
        sys.exit(0 if success else 1)
    
    # Default: test one voice
    print("Testing default voice (en-US-AriaNeural)...")
    success = await test_voice("en-US-AriaNeural", args.text)
    print("\nTip: Use --all to test all voices, --list to see configured voices")
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
