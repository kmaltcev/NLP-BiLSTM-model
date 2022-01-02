"""import json
with open('metadata.json') as fp:
    meta = json.load(fp)
    sorted_meta = dict(sorted(meta.items(), key=lambda item: item[1]))
    """

import matplotlib.pyplot as plt
import seaborn as sns

titanic_dataset = sns.load_dataset("titanic")

sns.barplot(x="class", y="survived", hue="embark_town", data=titanic_dataset)
plt.show()
