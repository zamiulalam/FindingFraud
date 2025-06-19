import json
import matplotlib.pyplot as plt
with open("importance.json") as f:
    data = json.load(f)

# Sort the dictionary by value in descending order
sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:30]
keys, values = zip(*sorted_items)

print(list(keys))
# # Create bar plot
# plt.figure(figsize=(10, 6))
# plt.bar(keys, values)
# plt.xlabel('Category')
# plt.ylabel('Importance (gain)')
# plt.title('Bar Plot of Dictionary Values (Descending Order)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()