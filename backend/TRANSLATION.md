# Translation Service Documentation

The Translation Service provides Chinese ↔ Min Nan translation capabilities using a hybrid approach combining Neural Machine Translation (NMT) and dictionary-based translation.

## Overview

The translation service uses:

1. **Neural Machine Translation (NMT)**: Facebook's M2M-100 model for high-quality translation
2. **Dictionary-based Translation**: Fast fallback for common phrases and words
3. **Hybrid Approach**: Combines both methods for optimal results

## Features

- ✅ Chinese (Mandarin/Traditional) to Min Nan translation
- ✅ Min Nan to Chinese translation
- ✅ Dictionary with 40+ common phrases
- ✅ Neural translation with M2M-100
- ✅ Automatic fallback to dictionary if NMT fails
- ✅ Post-processing for domain-specific terms

## Usage

### Basic Translation

```python
from app.services.translation_service import translation_service
from app.models.schemas import LanguageType

# Translate Chinese to Min Nan
translated_text, processing_time = translation_service.translate(
    text="你好",
    source_language=LanguageType.CHINESE,
    target_language=LanguageType.MIN_NAN
)
print(translated_text)  # Output: "汝好"
```

### Dictionary-only Translation (Faster)

```python
# Use dictionary only (no neural translation)
translated_text, processing_time = translation_service.translate(
    text="謝謝",
    source_language=LanguageType.CHINESE,
    target_language=LanguageType.MIN_NAN,
    use_neural=False
)
print(translated_text)  # Output: "多謝"
```

### Translation via API

#### Endpoint

The translation is automatically integrated into the TTS endpoint:

```http
POST /api/v1/tts/synthesize
Content-Type: application/json

{
  "text": "你好",
  "source_language": "chinese",
  "target_language": "min_nan"
}
```

The text will be automatically translated before being converted to speech.

## Translation Dictionary

The service includes a built-in dictionary with common Min Nan phrases:

| Chinese | Min Nan | English |
|---------|---------|---------|
| 你好 | 汝好 | Hello |
| 謝謝 | 多謝 | Thank you |
| 再見 | 再會 | Goodbye |
| 是 | 是 | Yes |
| 不是 | 毋是 | No |
| 好 | 好 | Good/OK |
| 我 | 我 | I/me |
| 你 | 汝 | You |
| 他/她 | 伊 | He/She |
| 今天 | 今仔日 | Today |
| 明天 | 明仔載 | Tomorrow |
| 什麼 | 啥物 | What |
| 哪裡 | 佗位 | Where |
| 怎麼 | 按怎 | How |
| 為什麼 | 為啥物 | Why |
| 吃飯 | 食飯 | Eat meal |
| 喝水 | 飲水 | Drink water |
| 對不起 | 歹勢 | Sorry |
| 沒關係 | 無要緊 | It's okay |

...and 20+ more phrases!

## Neural Translation Model

### Model Information

- **Model**: `facebook/m2m100_418M`
- **Architecture**: Multilingual encoder-decoder
- **Languages**: Supports 100 languages
- **Size**: 418M parameters (smaller, faster variant)

### Loading the Model

```python
from app.services.translation_service import translation_service

# Load NMT model
translation_service.load_models()

# Check if loaded
print(translation_service.models_loaded)  # True or False
```

### How It Works

1. **Tokenization**: Input text is tokenized for the source language
2. **Encoding**: Text is encoded into multilingual representations
3. **Decoding**: Decoded into target language
4. **Post-processing**: Dictionary terms are applied for domain-specific accuracy

## Translation Workflow

### Chinese → Min Nan

```
Input: "你好，這是一個測試"
  ↓
1. Check if translation needed (different languages)
  ↓
2. Use Neural Translation (if enabled and loaded)
   - Tokenize with M2M-100
   - Generate translation
  ↓
3. Apply Dictionary Post-processing
   - Replace known terms: "你好" → "汝好"
  ↓
Output: "汝好，這是一个測試"
```

### Min Nan → Chinese

```
Input: "汝好"
  ↓
1. Check dictionary for exact match
  ↓
2. If not found, use Neural Translation
  ↓
3. Word-by-word fallback if needed
  ↓
Output: "你好"
```

## Configuration

### Environment Variables

Add to `.env`:

```bash
# Translation model (optional, uses default if not set)
TRANSLATION_MODEL=facebook/m2m100_418M

# Device for translation
DEVICE=cuda  # or 'cpu'

# Model cache directory
MODEL_CACHE_DIR=./model_cache
```

### Programmatic Configuration

```python
from app.services.translation_service import TranslationService

service = TranslationService()

# Add custom dictionary entries
service.chinese_to_minnan_dict["新詞"] = "新詞的閩南語翻譯"
```

## Performance

### Translation Speed

| Method | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| Dictionary | ~1ms | High for known phrases | Common phrases |
| Neural (CPU) | ~500ms | High | Complex sentences |
| Neural (GPU) | ~100ms | High | Complex sentences |

### Memory Usage

- Dictionary only: ~10 MB
- With Neural model: ~500 MB - 2 GB

## Error Handling

The translation service gracefully handles errors:

```python
try:
    result, time = translation_service.translate(
        "你好",
        LanguageType.CHINESE,
        LanguageType.MIN_NAN
    )
except Exception as e:
    # Fallback to original text
    print(f"Translation failed: {e}")
```

### Automatic Fallbacks

1. Neural translation fails → Use dictionary
2. Dictionary has no match → Keep original text
3. Model not loaded → Dictionary-only mode

## Extending the Dictionary

### Adding New Phrases

```python
from app.services.translation_service import translation_service

# Add new translations
new_translations = {
    "早餐": "早頓",
    "午餐": "中晝",
    "晚餐": "暗頓"
}

translation_service.chinese_to_minnan_dict.update(new_translations)

# Rebuild reverse dictionary
translation_service.minnan_to_chinese_dict = {
    v: k for k, v in translation_service.chinese_to_minnan_dict.items()
}
```

### Loading from File

```python
import json

# Load dictionary from JSON file
with open('custom_dict.json', 'r', encoding='utf-8') as f:
    custom_dict = json.load(f)

translation_service.chinese_to_minnan_dict.update(custom_dict)
```

## Fine-tuning for Min Nan

The M2M-100 model doesn't natively support Min Nan, so results may vary. For production use, consider:

### Option 1: Fine-tune M2M-100

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, Trainer

# Prepare Min Nan parallel corpus
train_dataset = [
    {"chinese": "你好", "minnan": "汝好"},
    # ... more examples
]

# Fine-tune model
# (Detailed fine-tuning code here)
```

### Option 2: Use Dedicated Min Nan Model

If a Min Nan-specific model becomes available:

```python
# Update configuration
MINNAN_TRANSLATION_MODEL = "your-org/chinese-minnan-translator"

# Use in translation service
# (Update model loading code)
```

## Testing Translation

### Unit Tests

```bash
# Run translation tests
pytest tests/unit/test_translation_service.py -v

# Test dictionary only
pytest tests/unit/test_translation_service.py::TestTranslationService::test_dictionary_translate
```

### Manual Testing

```python
from app.services.translation_service import translation_service
from app.models.schemas import LanguageType

# Test translations
test_phrases = ["你好", "謝謝", "再見", "早安"]

for phrase in test_phrases:
    result, time = translation_service.translate(
        phrase,
        LanguageType.CHINESE,
        LanguageType.MIN_NAN,
        use_neural=False
    )
    print(f"{phrase} → {result} ({time:.3f}s)")
```

## API Examples

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/v1/tts/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，歡迎使用閩南語系統",
    "source_language": "chinese",
    "target_language": "min_nan"
  }'
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/tts/synthesize",
    json={
        "text": "你好",
        "source_language": "chinese",
        "target_language": "min_nan"
    }
)

result = response.json()
print(f"Translated: {result['text']}")
print(f"Audio URL: {result['audio_url']}")
```

## Best Practices

1. **Use Dictionary for Common Phrases**: Faster and more accurate
2. **Preload Models**: Load models at startup to avoid delays
3. **Handle Fallbacks**: Always have fallback handling
4. **Cache Results**: Consider caching frequent translations
5. **Monitor Performance**: Track translation times and accuracy

## Troubleshooting

### Model Download Issues

```bash
# Manually download model
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
```

### Translation Quality Issues

- Add more dictionary entries for your domain
- Consider fine-tuning the model
- Use post-processing rules
- Combine multiple translation approaches

### Performance Issues

- Use GPU if available
- Use smaller M2M model variant
- Cache frequent translations
- Use dictionary-only mode for simple phrases

## Future Improvements

- [ ] Fine-tune model on Min Nan parallel corpus
- [ ] Add more dictionary entries
- [ ] Support for different Min Nan romanization systems (POJ, Tâi-lô)
- [ ] Translation confidence scores
- [ ] Multiple translation candidates
- [ ] Context-aware translation
- [ ] Translation memory
- [ ] User feedback integration

## References

- [M2M-100 Paper](https://arxiv.org/abs/2010.11125)
- [Hugging Face M2M-100](https://huggingface.co/facebook/m2m100_418M)
- [Min Nan Wikipedia](https://zh-min-nan.wikipedia.org/)
- [POJ (Pe̍h-ōe-jī) System](https://en.wikipedia.org/wiki/Pe%CC%8Dh-%C5%8De-j%C4%AB)
