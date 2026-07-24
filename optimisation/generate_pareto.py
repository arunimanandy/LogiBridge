import pandas as pd

import matplotlib.pyplot as plt


df = pd.read_csv("results/benchmark_results.csv")


plt.figure(figsize=(8,6))


for _, row in df.iterrows():

    plt.scatter(

        row["model_size_kb"],

        row["p95_latency_ms"],

        s=120

    )


    plt.annotate(

        row["variant"],

        (row["model_size_kb"], row["p95_latency_ms"])

    )


plt.xlabel("Model Size (KB)")

plt.ylabel("p95 Latency (ms)")

plt.title("Pareto Chart - Model Size vs p95 Latency")

plt.grid(True)


plt.savefig(

    "results/pareto_chart.png",

    dpi=300,

    bbox_inches="tight"

)


print("Pareto chart created")