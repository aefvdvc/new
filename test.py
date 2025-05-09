import numpy as np
import matplotlib.pyplot as plt

nb_classes = 6
angles = np.linspace(0, 2*np.pi, nb_classes, endpoint=False)
means = np.array([(np.cos(theta), np.sin(theta)) for theta in angles])

plt.figure(figsize=(5,5))
plt.scatter(means[:,0], means[:,1], color='red', label='Cluster Centers')
for i, (x, y) in enumerate(means):
    plt.text(x, y, f'Class {i}', fontsize=12, ha='right')
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.gca().set_aspect('equal')
plt.title("Means of Gaussian Components in Polar Form")
plt.legend()
plt.show()
