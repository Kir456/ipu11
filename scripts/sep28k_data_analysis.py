import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_dataset():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    csv_path = base_dir / "data" / "sep28k_audio" / "metadata.csv"
    
    if not csv_path.exists():
        print(f"File {csv_path} not found!")
        return

    df = pd.read_csv(csv_path)
    
    print(f"Total audio fragments: {len(df)}")
    
    class_counts = df['is_stutter'].value_counts()
    class_percentages = df['is_stutter'].value_counts(normalize=True) * 100
    
    print("\n--- Binary class balance ---")
    print(f"Fluent speech (0): {class_counts.get(0, 0)} ({class_percentages.get(0, 0):.2f}%)")
    print(f"Stutter (1):     {class_counts.get(1, 0)} ({class_percentages.get(1, 0):.2f}%)")
    
    stutter_types = ['prolongation', 'block', 'sound_repetition', 'word_repetition', 'interjection']
    print("\n--- Occurrence of specific types (can overlap) ---")
    
    type_stats = {}
    for col in stutter_types:
        count = (df[col] > 0).sum()
        percentage = (count / len(df)) * 100
        type_stats[col] = count
        print(f"{col.ljust(20)}: {count} ({percentage:.2f}%)")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.countplot(data=df, x='is_stutter', ax=axes[0], palette=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Binary Class Balance (0 = Fluent, 1 = Stutter)', fontsize=14)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Number of fragments')
    
    types_s = pd.Series(type_stats).sort_values(ascending=False)
    sns.barplot(x=types_s.values, y=types_s.index, ax=axes[1], palette='viridis')
    axes[1].set_title('Distribution of Disfluency Types', fontsize=14)
    axes[1].set_xlabel('Number of fragments')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_dataset()