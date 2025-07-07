#!/usr/bin/env python3
"""
🌐 Translation Updater Demo - How to Add New Languages
Demonstrates various ways to use the translation updater for adding new languages

🎯 USAGE EXAMPLES:
1. Add single language to all models
2. Add multiple languages to specific models
3. Enable disabled languages
4. List available languages
"""

import os
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

def demo_list_available_languages():
    """Demo: List all available languages"""
    print("🌐 DEMO 1: List Available Languages")
    print("=" * 50)
    
    os.system("python scripts/translation_updater.py --list-available")
    print()

def demo_add_single_language():
    """Demo: Add Tamil to all models"""
    print("🌐 DEMO 2: Add Tamil to All Models")
    print("=" * 50)
    
    # First enable Tamil in config
    print("Step 1: Enable Tamil in configuration...")
    os.system("python scripts/translation_updater.py --enable-language ta")
    
    # Then add Tamil to all models
    print("\nStep 2: Add Tamil to all models...")
    os.system("python scripts/translation_updater.py --add-language ta --target-models all")
    print()

def demo_add_multiple_languages():
    """Demo: Add Tamil and Kannada to specific models"""
    print("🌐 DEMO 3: Add Tamil and Kannada to Specific Models")
    print("=" * 50)
    
    # Enable both languages
    print("Step 1: Enable languages in configuration...")
    os.system("python scripts/translation_updater.py --enable-language ta")
    os.system("python scripts/translation_updater.py --enable-language kn")
    
    # Add to specific models
    print("\nStep 2: Add to A_universal_full and B_universal_lite...")
    os.system("python scripts/translation_updater.py --add-languages ta,kn --target-models A_universal_full,B_universal_lite")
    print()

def demo_add_to_category_models():
    """Demo: Add languages to category-specific models"""
    print("🌐 DEMO 4: Add Languages to Category Models")
    print("=" * 50)
    
    # Add to healthcare category
    print("Adding Tamil to healthcare category...")
    os.system("python scripts/translation_updater.py --add-language ta --target-models C_category_specific/healthcare")
    print()

def show_usage_examples():
    """Show various usage examples"""
    print("🌐 TRANSLATION UPDATER USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        {
            "title": "List Available Languages",
            "command": "python scripts/translation_updater.py --list-available",
            "description": "Shows all languages that can be added"
        },
        {
            "title": "Enable Language in Config",
            "command": "python scripts/translation_updater.py --enable-language ta",
            "description": "Enables Tamil in the configuration file"
        },
        {
            "title": "Add Single Language to All Models",
            "command": "python scripts/translation_updater.py --add-language ta --target-models all",
            "description": "Adds Tamil to all existing model variants"
        },
        {
            "title": "Add Multiple Languages to Specific Models",
            "command": "python scripts/translation_updater.py --add-languages ta,kn --target-models A_universal_full,B_universal_lite",
            "description": "Adds Tamil and Kannada to specific model variants"
        },
        {
            "title": "Add Language to Category Models",
            "command": "python scripts/translation_updater.py --add-language hi --target-models C_category_specific",
            "description": "Adds Hindi to all category-specific models"
        },
        {
            "title": "Add Language to Specific Category",
            "command": "python scripts/translation_updater.py --add-language te --target-models C_category_specific/healthcare",
            "description": "Adds Telugu only to healthcare category model"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")
    
    print("\n" + "=" * 60)

def show_workflow():
    """Show recommended workflow for adding new languages"""
    print("🌐 RECOMMENDED WORKFLOW FOR ADDING NEW LANGUAGES")
    print("=" * 60)
    
    workflow_steps = [
        "1. Check available languages: --list-available",
        "2. Enable language in config: --enable-language <lang>",
        "3. Add to models: --add-language <lang> --target-models <targets>",
        "4. Verify integration with existing speech models",
        "5. Test translation pipeline functionality"
    ]
    
    for step in workflow_steps:
        print(f"   {step}")
    
    print("\n📋 BENEFITS OF STANDALONE UPDATER:")
    benefits = [
        "✅ No need to regenerate all speech models",
        "✅ Selective language addition to specific models",
        "✅ Preserves existing model configurations",
        "✅ Fast updates without full pipeline rerun",
        "✅ Easy rollback if needed"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")
    
    print("\n" + "=" * 60)

def main():
    """Main demo function"""
    print("🚀 TRANSLATION UPDATER DEMONSTRATION")
    print("=" * 60)
    
    # Show usage examples
    show_usage_examples()
    
    # Show recommended workflow
    show_workflow()
    
    # Interactive demo choice
    print("\n🎯 INTERACTIVE DEMO OPTIONS:")
    print("1. List available languages")
    print("2. Show how to add single language")
    print("3. Show how to add multiple languages")
    print("4. Show how to add to category models")
    print("5. Run all demos")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-5): ").strip()
    
    if choice == "1":
        demo_list_available_languages()
    elif choice == "2":
        demo_add_single_language()
    elif choice == "3":
        demo_add_multiple_languages()
    elif choice == "4":
        demo_add_to_category_models()
    elif choice == "5":
        print("\n🚀 Running all demos...")
        demo_list_available_languages()
        demo_add_single_language()
        demo_add_multiple_languages()
        demo_add_to_category_models()
    elif choice == "0":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main() 