import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from scipy.stats import f_oneway

def get_options():
    parser = argparse.ArgumentParser(description="Generate swarm plot with significance annotations and save ANOVA results.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the swarm plot.")
    parser.add_argument("-as", "--anova_save", required=True, help="Path to save ANOVA results.")
    parser.add_argument("-cs", "--circle_size", type=float, default=5.0, help="Swarm plot circle size.")
    parser.add_argument("-cp", "--color_palette", default="Set2", help="Color palette for plots.")
    parser.add_argument("-w", "--width", type=int, default=12, help="Plot width in inches.")
    parser.add_argument("-ht", "--height", type=int, default=8, help="Plot height in inches.")
    parser.add_argument("-r", "--rotation", type=int, default=45, help="X-axis label rotation.")
    return parser.parse_args()

def main():
    args = get_options()

    # Load the dataset
    data = pd.read_csv(args.input)

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Generate swarm plot
    plt.figure(figsize=(args.width, args.height))
    ax = sns.swarmplot(
        x="Strain",
        y="Score",
        data=data,
        palette=args.color_palette,
        size=args.circle_size
    )

    # Perform ANOVA for each pair and save results
    grouped = data.groupby('Strain')['Score'].apply(list)
    f_stat, p_value = f_oneway(*grouped)
    
    # Save overall ANOVA results
    with open(args.anova_save, 'w') as f:
        f.write("Overall ANOVA Results:\n")
        f.write(f"F-statistic: {f_stat}\n")
        f.write(f"p-value: {p_value}\n\n")
        f.write("Pairwise Comparisons:\n")

        # Pairwise comparisons
        unique_strains = data["Strain"].unique()
        pairs = [(unique_strains[i], unique_strains[j]) for i in range(len(unique_strains)) for j in range(i + 1, len(unique_strains))]
        pairwise_pvalues = []

        for strain1, strain2 in pairs:
            if strain1 in grouped and strain2 in grouped:
                _, pair_pvalue = f_oneway(grouped[strain1], grouped[strain2])
                f.write(f"{strain1} vs {strain2}: p-value = {pair_pvalue}\n")
                pairwise_pvalues.append(pair_pvalue)

    # Annotate significance
    annotator = Annotator(ax, pairs, data=data, x="Strain", y="Score")
    annotator.set_pvalues(pairwise_pvalues)
    annotator.configure(line_width=1, text_format="star", loc="outside")
    annotator.annotate()

    # Customize and save plot
    ax.set_xlabel("Strain", fontsize=12)
    ax.set_ylabel("Fitness Score", fontsize=12)
    plt.xticks(rotation=args.rotation, ha='right')
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')
    print(f"Swarm plot saved to '{args.output}'.")
    print(f"ANOVA results saved to '{args.anova_save}'.")
    plt.close()

if __name__ == "__main__":
    main()
