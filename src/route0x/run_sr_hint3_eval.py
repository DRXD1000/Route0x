from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.encoders import CohereEncoder
from semantic_router.layer import RouteLayer

import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import time
# import seaborn as sns



dataset = "rostd"
model = "mpnet"
version = f"sr_v1"
benchmark_file = f"./eval/{dataset}/fs_benchmark_{version}_model_{model}.csv"
ood_label = "NO_NODES_DETECTED"
test_file = "./data/rostd_test.csv"


file_path = 'generated_datasets/synthetic_gpt-4o_personal_assistant_20241101_214133_train.csv'
df = pd.read_csv(file_path)
df['label'] = df['label'].apply(lambda x: x.title() if x != ood_label else x)

routes_dict = {}

for _, row in df.iterrows():
    route_name = row['label'] 
    utterance = row['text']   

    if route_name not in routes_dict:
        routes_dict[route_name] = []

    routes_dict[route_name].append(utterance)

routes = []
for route_name, utterances in routes_dict.items():
    route = Route(
        name=route_name,
        utterances=utterances
    )
    routes.append(route)



# os.environ["COHERE_API_KEY"] = "O84zGUhxFKwYj9O0xXtnRHaH2cTTHFH1ZE7FfdWX"
# encoder = CohereEncoder()
encoder = HuggingFaceEncoder(name="sentence-transformers/all-mpnet-base-v2")
# encoder = HuggingFaceEncoder(name="BAAI/bge-base-en-v1.5")

rl = RouteLayer(encoder=encoder, routes=routes)

id_correct = 0
id_incorrect = 0

ood_correct = 0
ood_incorrect = 0

total_in_domain = 0
total_out_domain = 0

if os.path.exists(benchmark_file):
    print(f"{benchmark_file} exists. Loading predictions from the file.")
    results_df = pd.read_csv(benchmark_file)

    for index, row in results_df.iterrows():
        true_label = row['label']
        predicted_route = row['SR route']
        prob = row['prob']  

        if true_label == ood_label:
            total_out_domain += 1

            if predicted_route == ood_label:
                ood_correct += 1
            else:
                ood_incorrect += 1

        else:
            total_in_domain += 1
            if predicted_route == true_label:
                id_correct += 1
            else:
                id_incorrect += 1

else:
    
    df = pd.read_csv(test_file)
    df['label'] = df['label'].apply(lambda x: x.title() if x != ood_label else x)
    results = []
    count = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        count += 1
        sample_text = row['text']
        if sample_text == "" or pd.isna(sample_text):
            continue
        true_label = row['label']

        predicted_route = rl(sample_text).name
        if not predicted_route:
            predicted_route = ood_label
        prob = "NA"

        if true_label == ood_label:
            total_out_domain += 1

            if predicted_route == ood_label:
                ood_correct += 1
            else:
                ood_incorrect += 1

        else:
            total_in_domain += 1
            if predicted_route == true_label:
                id_correct += 1
            else:
                id_incorrect += 1
       

        
        results.append({
            'text': sample_text,
            'label': true_label,
            'SR route': predicted_route,
            'prob':  prob
        })

        if model == "cohere" and count % 90 == 0:
            time.sleep(15)  

    # Save the predictions to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(benchmark_file, index=False)
    print(f"Benchmark file created: {benchmark_file}")


# Print classification report
print("Classification Report:\n")
report = classification_report(results_df['label'], results_df['SR route'])
print(report)
with open(f'./eval/{dataset}/{version}_{model}_creport.txt', 'w') as file:
    file.write(report)


# Calculate percentages for ID, OOD accuracy, and ID OOS incorrect
id_correct_pct = (id_correct / total_in_domain) * 100 if total_in_domain > 0 else 0
ood_correct_pct = (ood_correct / total_out_domain) * 100 if total_out_domain > 0 else 0

# Group the bars together for ID, OOD, and ID OOS incorrect accuracy
plt.figure(figsize=(8, 6))

# Setting up the bars
bar_width = 0.25
index = np.arange(1)  # Only one group, so index is [0]
bars1 = plt.bar(index, id_correct_pct, bar_width, label='True ID', color='#4CAF50')
bars2 = plt.bar(index + bar_width, ood_correct_pct, bar_width, label='ID OOS', color='#FFC107')  

# Add percentage labels on top of bars with white text for better visibility
for bar, pct in zip(bars1, [id_correct_pct]):
    plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() - 5, f'{pct:.2f}%', ha='center', va='bottom', color='white', fontweight='bold')

for bar, pct in zip(bars2, [ood_correct_pct]):
    plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() - 5, f'{pct:.2f}%', ha='center', va='bottom', color='white', fontweight='bold')

# Customize the plot
plt.ylim(0, 100)  # Y-axis limit for 100%
plt.ylabel('Percentage of Correct Predictions', fontsize=12)
plt.title('In-Domain vs Out-of-Domain Correct Predictions', fontsize=14)
plt.xticks(index + bar_width, ['Predictions'], fontsize=12)
plt.yticks(fontsize=12)
plt.legend()

# Save the bar chart as a PNG file
output_file = f"./eval/{dataset}/fs_comparison_{version}_{model}.png"
plt.tight_layout()
plt.savefig(output_file)
print(f"Bar chart saved to {output_file}")

