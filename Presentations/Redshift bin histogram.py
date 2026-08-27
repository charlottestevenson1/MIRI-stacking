import matplotlib.pyplot as plt

files = [
    ("z=8–9", "Redshift Bins/Bin objects/Redshifts 8-9.txt"),
    ("z=9–10", "Redshift Bins/Bin objects/Redshifts 9-10.txt"),
    ("z=10–11", "Redshift Bins/Bin objects/Redshifts 10-11.txt"),
    ("z=11–12", "Redshift Bins/Bin objects/Redshifts 11-12.txt"),
    ("z=12–15", "Redshift Bins/Bin objects/Redshifts 12-15.txt"),
]

labels = []
counts = []

for label, filename in files:
    with open(filename) as f:
        galaxies = [line.strip() for line in f if line.strip()]

    labels.append(label)
    counts.append(len(galaxies))

fig, ax = plt.subplots(figsize=(12, 7), facecolor="none")
ax.set_facecolor("none")

bars = ax.bar(
    labels,
    counts,
    color="steelblue",
    edgecolor="white",
    linewidth=1.5,
    alpha=0.8
)

# Number above each bar
for bar, count in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        str(count),
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color="white"
    )

ax.set_xlabel(
    "Redshift bin",
    fontsize=20,
    color="white",
    labelpad=12
)

ax.set_ylabel(
    "Number of galaxies",
    fontsize=20,
    color="white",
    labelpad=12
)

ax.tick_params(
    axis="both",
    colors="white",
    labelsize=16,
    width=1.5,
    length=6
)

for spine in ax.spines.values():
    spine.set_color("white")
    spine.set_linewidth(1.5)

ax.grid(
    axis="y",
    color="white",
    alpha=0.15
)

plt.tight_layout()

plt.savefig(
    "Presentations/galaxies_per_redshift_bin.png",
    dpi=400,
    bbox_inches="tight",
    transparent=True
)

plt.show()