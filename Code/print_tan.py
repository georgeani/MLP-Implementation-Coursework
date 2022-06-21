from backpropagation_algorithm import output_tan, output_derivative_tan
import numpy as np
import matplotlib.pyplot as plt

man = list()
deriv = list()
deriv2 = list()


def generate_sample_data(start, end, step):
    # Generates sample data using np.linspace
    return np.linspace(start, end, step)


data = generate_sample_data(-5, 5, 10)
for d in data:
    man.append(output_tan(d))

nu = np.tanh(data)

for de in range(len(man)):
    deriv.append(output_derivative_tan(man[de]))
    deriv2.append(output_derivative_tan(nu[de]))

plt.plot(man)
plt.show()

plt.plot(deriv)
plt.show()

plt.plot(deriv2)
plt.show()

plt.plot(nu)
plt.show()
