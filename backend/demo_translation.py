#!/usr/bin/env python3
"""
Demo script to test the translation service

This script demonstrates the Chinese ↔ Min Nan translation functionality
without requiring the full server to be running.
"""

import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.dirname(__file__))

from app.services.translation_service import TranslationService
from app.models.schemas import LanguageType


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def test_dictionary_translation():
    """Test dictionary-based translation"""
    print_header("Dictionary Translation Tests")

    service = TranslationService()

    test_phrases = [
        ("你好", LanguageType.CHINESE, LanguageType.MIN_NAN),
        ("謝謝", LanguageType.CHINESE, LanguageType.MIN_NAN),
        ("再見", LanguageType.CHINESE, LanguageType.MIN_NAN),
        ("早安", LanguageType.CHINESE, LanguageType.MIN_NAN),
        ("對不起", LanguageType.CHINESE, LanguageType.MIN_NAN),
        ("我", LanguageType.CHINESE, LanguageType.MIN_NAN),
        ("你", LanguageType.CHINESE, LanguageType.MIN_NAN),
        ("今天", LanguageType.CHINESE, LanguageType.MIN_NAN),
        ("明天", LanguageType.CHINESE, LanguageType.MIN_NAN),
        ("什麼", LanguageType.CHINESE, LanguageType.MIN_NAN),
    ]

    print("Chinese → Min Nan (Dictionary)\n")

    for text, src, tgt in test_phrases:
        result, time_taken = service.translate(text, src, tgt, use_neural=False)
        print(f"  {text:12} → {result:12} ({time_taken*1000:.2f}ms)")

    # Reverse translation
    print("\nMin Nan → Chinese (Dictionary)\n")

    reverse_phrases = [
        ("汝好", LanguageType.MIN_NAN, LanguageType.CHINESE),
        ("多謝", LanguageType.MIN_NAN, LanguageType.CHINESE),
        ("再會", LanguageType.MIN_NAN, LanguageType.CHINESE),
    ]

    for text, src, tgt in reverse_phrases:
        result, time_taken = service.translate(text, src, tgt, use_neural=False)
        print(f"  {text:12} → {result:12} ({time_taken*1000:.2f}ms)")


def test_neural_translation():
    """Test neural machine translation"""
    print_header("Neural Machine Translation Tests")

    service = TranslationService()

    print("Loading M2M-100 translation model...")
    print("(This may take a few minutes on first run)\n")

    try:
        service.load_models()
        print("✅ Model loaded successfully!\n")

        test_sentences = [
            "你好，這是一個測試",
            "我喜歡學習閩南語",
            "今天天氣很好",
        ]

        print("Chinese → Min Nan (Neural Translation)\n")

        for text in test_sentences:
            result, time_taken = service.translate(
                text,
                LanguageType.CHINESE,
                LanguageType.MIN_NAN,
                use_neural=True
            )
            print(f"  Input:  {text}")
            print(f"  Output: {result}")
            print(f"  Time:   {time_taken:.2f}s\n")

    except Exception as e:
        print(f"⚠️  Neural translation not available: {str(e)}")
        print("   Using dictionary-based translation as fallback\n")

        for text in ["你好", "謝謝"]:
            result, time_taken = service.translate(
                text,
                LanguageType.CHINESE,
                LanguageType.MIN_NAN,
                use_neural=False
            )
            print(f"  {text} → {result} ({time_taken*1000:.2f}ms)")


def test_hybrid_translation():
    """Test hybrid approach (neural + dictionary)"""
    print_header("Hybrid Translation (Neural + Dictionary Post-processing)")

    service = TranslationService()

    if not service.models_loaded:
        try:
            service.load_models()
        except Exception as e:
            print(f"⚠️  Skipping hybrid test: {str(e)}\n")
            return

    test_text = "你好，我想要學習閩南語"

    print("Input:  ", test_text)
    print("\nProcessing steps:")
    print("  1. Neural translation with M2M-100")
    print("  2. Dictionary post-processing for known terms\n")

    result, time_taken = service.translate(
        test_text,
        LanguageType.CHINESE,
        LanguageType.MIN_NAN,
        use_neural=True
    )

    print(f"Output: {result}")
    print(f"Time:   {time_taken:.2f}s")


def show_dictionary():
    """Show available dictionary entries"""
    print_header("Available Dictionary Entries")

    service = TranslationService()

    print(f"Total entries: {len(service.chinese_to_minnan_dict)}\n")
    print("Sample entries (Chinese → Min Nan):\n")

    # Show first 20 entries
    for i, (chinese, minnan) in enumerate(service.chinese_to_minnan_dict.items()):
        if i < 20:
            print(f"  {chinese:12} → {minnan}")
        else:
            break

    print(f"\n  ... and {len(service.chinese_to_minnan_dict) - 20} more!")


def interactive_mode():
    """Interactive translation mode"""
    print_header("Interactive Translation Mode")

    service = TranslationService()

    print("Enter Chinese text to translate to Min Nan")
    print("(Type 'quit' to exit)\n")

    while True:
        try:
            text = input("Chinese: ").strip()

            if text.lower() == 'quit':
                break

            if not text:
                continue

            # Try dictionary first
            result_dict, time_dict = service.translate(
                text,
                LanguageType.CHINESE,
                LanguageType.MIN_NAN,
                use_neural=False
            )

            print(f"Min Nan (dictionary): {result_dict} ({time_dict*1000:.2f}ms)")

            # If models loaded, try neural too
            if service.models_loaded:
                result_neural, time_neural = service.translate(
                    text,
                    LanguageType.CHINESE,
                    LanguageType.MIN_NAN,
                    use_neural=True
                )
                print(f"Min Nan (neural):     {result_neural} ({time_neural:.2f}s)")

            print()

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}\n")


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("  Min Nan & Chinese Translation Demo")
    print("=" * 60)

    print("\nSelect option:")
    print("  1. Test dictionary translation")
    print("  2. Test neural translation")
    print("  3. Test hybrid translation")
    print("  4. Show dictionary entries")
    print("  5. Interactive mode")
    print("  6. Run all tests")
    print("  0. Exit")

    try:
        choice = input("\nEnter option (0-6): ").strip()

        if choice == "1":
            test_dictionary_translation()
        elif choice == "2":
            test_neural_translation()
        elif choice == "3":
            test_hybrid_translation()
        elif choice == "4":
            show_dictionary()
        elif choice == "5":
            interactive_mode()
        elif choice == "6":
            test_dictionary_translation()
            test_neural_translation()
            test_hybrid_translation()
            show_dictionary()
        elif choice == "0":
            print("\nGoodbye!")
            return
        else:
            print("\n❌ Invalid option")

        print("\n" + "=" * 60)
        print("  Demo Complete!")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
