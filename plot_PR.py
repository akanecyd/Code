import matplotlib.pyplot as plt
import json

pr_value = json.load(open('k1565_values.json', 'r'))

recall = pr_value['recall']
precision = pr_value['precision']
plt.plot(recall, precision, linewidth=2)
plt.gca().set_aspect('equal')
plt.xlim([0, 1])
plt.xlabel('Recall')
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.show()
