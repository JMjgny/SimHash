from freqvssal import compare_frequency_vs_saliency

results = compare_frequency_vs_saliency(
    r"C:\Users\Marc\.cache\kagglehub\datasets\labid93\image-forgery-detection\versions\1\Dataset\Original\654.jpg",
    r"C:\Users\Marc\.cache\kagglehub\datasets\labid93\image-forgery-detection\versions\1\Dataset\Forged\bright_new.jpg"
)

print(results)