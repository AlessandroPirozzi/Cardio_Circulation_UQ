import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

labels = ['R_AR_SYS', 'C_AR_SYS']
num_parameters = len(labels)
x_100 = []
x_500 = []
x_1000 = []
x_2000 = []
x_ref = []

with open("sa_2") as file_name:
    W = np.loadtxt(file_name, delimiter=",")
for i in range(num_parameters):
    x_100.append(W[0][i])
    x_500.append(W[1][i])
    x_1000.append(W[2][i])
    x_2000.append(W[3][i])
    x_ref.append(W[4][i])

color1 = 'steelblue'
color2 = 'red'
line_size = 4

df = pd.DataFrame({'Sobol index': x_100})
df.index = labels
df = df.sort_values(by='Sobol index', ascending=False)

df['Cumulative percentage'] = df['Sobol index'].cumsum()/df['Sobol index'].sum()*100

fig1, ax1 = plt.subplots(figsize=(8, 7))
ax1.bar(df.index, df['Sobol index'], color=color1)
ax2 = ax1.twinx()
ax2.plot(df.index, df['Cumulative percentage'], color=color2, marker="D", ms=line_size)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax1.set_xticklabels(df.index, rotation=35)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
fig1.savefig('pareto_2_100.png')

df = pd.DataFrame({'Sobol index': x_500})
df.index = labels
df = df.sort_values(by='Sobol index', ascending=False)

df['Cumulative percentage'] = df['Sobol index'].cumsum()/df['Sobol index'].sum()*100

fig2, ax3 = plt.subplots(figsize=(8, 7))
ax3.bar(df.index, df['Sobol index'], color=color1)
ax4 = ax3.twinx()
ax4.plot(df.index, df['Cumulative percentage'], color=color2, marker="D", ms=line_size)
ax4.yaxis.set_major_formatter(PercentFormatter())
ax3.set_xticklabels(df.index, rotation=35)
ax3.tick_params(axis='x', labelsize=12)
ax3.tick_params(axis='y', labelsize=12)
ax4.tick_params(axis='y', labelsize=12)
fig2.savefig('pareto_2_500.png')

df = pd.DataFrame({'Sobol index': x_1000})
df.index = labels
df = df.sort_values(by='Sobol index', ascending=False)

df['Cumulative percentage'] = df['Sobol index'].cumsum()/df['Sobol index'].sum()*100

fig3, ax5 = plt.subplots(figsize=(8, 7))
ax5.bar(df.index, df['Sobol index'], color=color1)
ax6 = ax5.twinx()
ax6.plot(df.index, df['Cumulative percentage'], color=color2, marker="D", ms=line_size)
ax6.yaxis.set_major_formatter(PercentFormatter())
ax5.set_xticklabels(df.index, rotation=35)
ax5.tick_params(axis='x', labelsize=12)
ax5.tick_params(axis='y', labelsize=12)
ax6.tick_params(axis='y', labelsize=12)
fig3.savefig('pareto_2_1000.png')

df = pd.DataFrame({'Sobol index': x_2000})
df.index = labels
df = df.sort_values(by='Sobol index', ascending=False)

df['Cumulative percentage'] = df['Sobol index'].cumsum()/df['Sobol index'].sum()*100

fig4, ax7 = plt.subplots(figsize=(8, 7))
ax7.bar(df.index, df['Sobol index'], color=color1)
ax8 = ax7.twinx()
ax8.plot(df.index, df['Cumulative percentage'], color=color2, marker="D", ms=line_size)
ax8.yaxis.set_major_formatter(PercentFormatter())
ax7.set_xticklabels(df.index, rotation=35)
ax7.tick_params(axis='x', labelsize=12)
ax7.tick_params(axis='y', labelsize=12)
ax8.tick_params(axis='y', labelsize=12)
fig4.savefig('pareto_2_2000.png')

df = pd.DataFrame({'Sobol index': x_ref})
df.index = labels
df = df.sort_values(by='Sobol index', ascending=False)

df['Cumulative percentage'] = df['Sobol index'].cumsum()/df['Sobol index'].sum()*100

fig5, ax9 = plt.subplots(figsize=(8, 7))
ax9.bar(df.index, df['Sobol index'], color=color1)
ax10 = ax9.twinx()
ax10.plot(df.index, df['Cumulative percentage'], color=color2, marker="D", ms=line_size)
ax10.yaxis.set_major_formatter(PercentFormatter())
ax9.set_xticklabels(df.index, rotation=35)
ax9.tick_params(axis='x', labelsize=12)
ax9.tick_params(axis='y', labelsize=12)
ax10.tick_params(axis='y', labelsize=12)
fig5.savefig('pareto_2_ref.png')
'''
size = 0.25
colors = plt.get_cmap('twilight')(np.linspace(0.2, 0.7, len(x_ref)))

fig1, ax1 = plt.subplots(figsize=(5, 5))
ax1.pie(x_100, radius=1, colors=colors, wedgeprops=dict(width=size, edgecolor='w'), normalize=True)
ax1.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.axis('off')
fig1.savefig('sa_2_100.png')

fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.pie(x_500, radius=1, colors=colors, wedgeprops=dict(width=size, edgecolor='w'), normalize=True)
ax2.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.axis('off')
fig2.savefig('sa_2_500.png')

fig3, ax3 = plt.subplots(figsize=(5, 5))
ax3.pie(x_1000, radius=1, colors=colors, wedgeprops=dict(width=size, edgecolor='w'), normalize=True)
ax3.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.axis('off')
fig3.savefig('sa_2_1000.png')

fig4, ax4 = plt.subplots(figsize=(5, 5))
ax4.pie(x_2000, radius=1, colors=colors, wedgeprops=dict(width=size, edgecolor='w'), normalize=True)
ax4.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.axis('off')
fig4.savefig('sa_2_2000.png')

fig5, ax5 = plt.subplots(figsize=(5, 5))
ax5.pie(x_ref, radius=1, colors=colors, wedgeprops=dict(width=size, edgecolor='w'), normalize=True)
ax5.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.axis('off')
fig5.savefig('sa_2_ref.png')
'''
plt.show()