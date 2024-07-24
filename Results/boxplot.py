# creating boxplot

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
boxData = pd.read_csv("LGG_GE_data_boxplotinput.csv")
boxData.head()

sns.catplot(x="Group", y="log2 (RSEM +1)", hue="Y", kind="box", data=boxData)

fig, ax = plt.subplots(figsize=(37, 25), dpi=300)
bp =sns.boxplot(x='Group', y='log2 (RSEM +1)', 
                  data=boxData,hue='Y',palette=['g', 'r'])
bp =sns.boxplot(y='log2 (RSEM +1)', x='Group', 
                 data=boxData, 
                  hue='Y', palette=['g', 'r'])
ax.set(ylim=(-2, 24))
sns.set(font_scale=5, style='white')
plt.title('')
handles, labels = bp.get_legend_handles_labels()
# specify just one legend
l = plt.legend(handles[0:2], labels[0:2])
fig.savefig('LGG_GE_boxplot.tif', dpi=300)
