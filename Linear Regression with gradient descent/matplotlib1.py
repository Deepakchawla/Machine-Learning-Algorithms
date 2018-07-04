from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


x, y = np.loadtxt('data.csv', delimiter=',', unpack=True)
x1 = [3.2, 3.1, 3.4, 4.4, 3.8, 3.5, 3.4, 4.1, 3.5, 3.0, 3.5, 3.0, 2.0, 2.9, 3.1, 2.8, 2.5, 3.0, 3.1, 2.5, 2.7, 2.9, 2.5, 3.0, 2.5, 2.9, 3.2, 3.0, 3.8, 3.2, 3.0, 2.8, 3.8, 2.7, 3.2, 3.0]
y1 = [1.8, 1.9, 1.8, 2.2, 2.2, 2.0, 2.0, 2.0, 2.1, 1.7, 1.9, 1.8, 1.9, 2.3, 2.5, 2.3, 2.4, 2.0, 2.5, 2.1, 2.1, 2.2, 1.9, 2.9, 1.9, 2.8, 2.5, 2.5, 2.9, 2.7, 2.3, 2.8, 3.0, 2.2, 2.6, 2.5]
plt.title("demo")
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.scatter(x,y, color='r', label='testing line', linewidth=1)
plt.plot(x1,y1, color='r', label='testing line', linewidth=1)
plt.legend()
plt.show()



