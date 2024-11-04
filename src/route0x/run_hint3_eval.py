import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from route_finder import RouteFinder
import seaborn as sns
import time
from typing import Callable, Any
from collections import defaultdict
import statistics



prob_threshold = 0.5
con = 10
nn = 10
msamples = 12
oos = "yes_aug_add_gen"
base = "mpnet_pa_arc_fl"
gen_samples = 10
dataset = "rostd"
os.makedirs(f"./eval/{dataset}", exist_ok=True)
version = f"oai_v1b_c{con}_nn{nn}_ms{msamples}_oos_{oos}_gen{gen_samples}_thresh{prob_threshold}_model_{base}"
benchmark_file = f"./eval/{dataset}/fs_benchmark_{version}.csv"
# route_template = "This belongs to {}"
route_template = "{}"
ood_label = "NO_NODES_DETECTED"
test_file = "./data/rostd_test.csv"
model_path = "./output/run_20241101_214344/route360_model/"


id_correct = 0
id_incorrect = 0
ood_correct = 0
ood_incorrect = 0
total_in_domain = 0
total_out_domain = 0
true_id_conf = []
ood_conf = []  


class QueryTimer:
    def __init__(self):
        self.latencies = []
        self.latencies_by_type = defaultdict(list)
        
    def wrap_query(self, func: Callable, *args, query_type: str = "default", **kwargs):
        """
        Wraps a query function and measures its execution time.
        
        Args:
            func: The query function to measure
            *args: Arguments to pass to the query function
            query_type: String identifier for the type of query (default: "default")
            **kwargs: Keyword arguments to pass to the query function
            
        Returns:
            tuple: (query_result, latency_in_seconds)
        """
        start_time = time.perf_counter()  # More precise than time.time()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        self.latencies.append(latency)
        self.latencies_by_type[query_type].append(latency)
        
        return result, latency
    
    def get_statistics(self, query_type: str = None) -> dict:
        """
        Calculate statistics for all queries or a specific query type.
        
        Args:
            query_type: Optional string identifier for specific query type
            
        Returns:
            dict: Statistics including mean, median, p90, etc.
        """
        latencies = self.latencies_by_type[query_type] if query_type else self.latencies
        
        if not latencies:
            return {"error": "No queries recorded"}
            
        sorted_latencies = sorted(latencies)
        p50_index = int(len(sorted_latencies) * 0.50)
        p99_index = int(len(sorted_latencies) * 0.99)
        
        return {
            "total_queries": len(latencies),
            "mean_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "p99_latency": sorted_latencies[p99_index],
            "p50_latency": sorted_latencies[p50_index],
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "total_time": sum(latencies)
        }


timer = QueryTimer()


if os.path.exists(benchmark_file):
    print(f"{benchmark_file} exists. Loading predictions from the file.")
    results_df = pd.read_csv(benchmark_file)

    for index, row in results_df.iterrows():
        true_label = row['label']
        predicted_route = row['Route42 route']
        prob = row['prob']  

        if true_label == ood_label:
            total_out_domain += 1

            if predicted_route == ood_label:
                ood_correct += 1
                ood_conf.append(prob)
            else:
                ood_incorrect += 1
        else:
            total_in_domain += 1
            if predicted_route == true_label:
                true_id_conf.append(prob)
                id_correct += 1
            else:
                id_incorrect += 1


else:
    
    router = RouteFinder(model_path, 
                         use_calibrated_head=True, 
                         return_raw_scores=False
    )
                        #  model_uncertainity_threshold_for_using_nn = 0.2)
    df = pd.read_csv(test_file)
    df['label'] = df['label'].apply(lambda x: x.title() if x != ood_label else x)
    results = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        sample_text = row['text']
        if sample_text == "" or pd.isna(sample_text):
            continue
        true_label = row['label']

        if true_label != ood_label:
            true_label = route_template.format(true_label)


        route_obj, latency = timer.wrap_query(router.find_route, sample_text)
        predicted_route = route_obj["route_name"]
        prob = route_obj['prob']  

        # if route_obj['is_outlier']:
        #     if prob >= 0.9:
        #         pass
        #     else:
        #         predicted_route = ood_label
        # else:
        #     if float(route_obj['prob']) <= prob_threshold:
        #         predicted_route = route_obj["majority_voted_route"]
        #         prob = route_obj["mean_distance_from_majority_route"]
            
        
        if true_label == ood_label:
            total_out_domain += 1

            if predicted_route == ood_label:
                ood_correct += 1
                ood_conf.append(prob)
            else:
                ood_incorrect += 1
        else:
            total_in_domain += 1
            if predicted_route == true_label:
                true_id_conf.append(prob)
                id_correct += 1
            else:
                id_incorrect += 1

        
        results.append({
            'text': sample_text,
            'label': true_label,
            'Route42 route': predicted_route,
            'pred label': route_obj['route_name'],
            'is_outlier': route_obj['is_outlier'],
            'prob':  prob,
            'most_nn': route_obj["majority_voted_route"],
            'most_nn_mean_dist': route_obj["mean_distance_from_majority_route"],
            'latency': latency
        })

    # Save the predictions to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(benchmark_file, index=False)
    print(f"Benchmark file created: {benchmark_file}")

print(timer.get_statistics())
# Plotting the area chart for OOD and ID
plt.figure(figsize=(10, 6))

# Create a DataFrame for ID and OOD
df = pd.DataFrame({
    'Confidence': np.concatenate([ood_conf, true_id_conf]),
    'Category': ['OOS Intents']*len(ood_conf) + ['ID Intents']*len(true_id_conf)
})

# Sort the confidence values to make the area chart smooth
df = df.sort_values(by='Confidence')

# Plot the area chart for OOD and True ID
sns.kdeplot(data=df, x='Confidence', hue='Category', fill=True, common_norm=False, palette={'OOS Intents': 'red', 'ID Intents': 'green'})


# Add labels and title
plt.title("Confidence Distribution for Out of Scope (ID and OOD) and In-Domain Intents")
plt.xlabel("Model's Confidence (0-100%)")
plt.ylabel("Density of Predictions")

# Annotating OOD rejection and ID acceptance regions
plt.text(0.2, 3.0, "Low Conf when Rejecting Intents", horizontalalignment='center', color='black', fontsize=10, bbox=dict(facecolor='lightgray', alpha=0.5))
plt.text(0.5, 3.0, "High Conf when Accepting Intents", horizontalalignment='center', color='black', fontsize=10, bbox=dict(facecolor='lightgray', alpha=0.5))

# Save the chart as a PNG file
output_file = f"./eval/{dataset}/fs_conf_dist_{version}.png"
plt.tight_layout()
plt.savefig(output_file)
print(f"Area chart saved to {output_file}")


# Print classification report
print("Classification Report:\n")
report = classification_report(results_df['label'], results_df['Route42 route'])
print(report)
with open(f'./eval/{dataset}/{version}_creport.txt', 'w') as file:
    file.write(report)


# Calculate percentages for ID and OOD accuracy
id_correct_pct = (id_correct / total_in_domain) * 100 if total_in_domain > 0 else 0
ood_correct_pct = (ood_correct / total_out_domain) * 100 if total_out_domain > 0 else 0

# Group the bars together for ID and OOD accuracy
plt.figure(figsize=(8, 6))

# Setting up the bars
bar_width = 0.35
index = np.arange(1)  # Only one group, so index is [0]
bars1 = plt.bar(index, id_correct_pct, bar_width, label='In-Domain', color='#4CAF50')
bars2 = plt.bar(index + bar_width, ood_correct_pct, bar_width, label='Out-of-Domain', color='#F44336')

# Add percentage labels on top of bars with white text for better visibility
for bar, pct in zip(bars1, [id_correct_pct]):
    plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() - 5, f'{pct:.2f}%', ha='center', va='bottom', color='white', fontweight='bold')

for bar, pct in zip(bars2, [ood_correct_pct]):
    plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() - 5, f'{pct:.2f}%', ha='center', va='bottom', color='white', fontweight='bold')


# Customize the plot
plt.ylim(0, 100)  # Y-axis limit for 100%
plt.ylabel('Percentage of Correct Predictions', fontsize=12)
plt.title('In-Domain vs Out-of-Domain Correct Predictions', fontsize=14)
plt.xticks(index + bar_width / 2, ['Predictions'], fontsize=12)
plt.yticks(fontsize=12)
plt.legend()

# Save the bar chart as a PNG file
output_file = f"./eval/{dataset}/fs_comparison_{version}.png"
plt.tight_layout()
plt.savefig(output_file)
print(f"Bar chart saved to {output_file}")







