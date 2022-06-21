import math
import random
import csv_reader
import matplotlib.pyplot as plt


# The libraries used by the algorithm as explained in the cw
# The main network initialization function, it is in charge of the forward pass and backwards pass
# the input names are what they denote: hidden = hidden layer nodes
# train is the training dataset and validation is the validation dataset
# tanh denotes whether or not to use the tanh activation formula
def network_ini(inputs, hidden, train, validation, testing, epochs, learning_parameter, tanh):
    # initializing the arrays used to keep the weights of each node as well as some error metrics
    hidden_nodes = list()
    rmse_value = list()
    msre_values = list()
    observed = list()
    real = list()
    error = list()

    # initializing the weights for the hidden nodes
    for h in range(hidden):
        weights = list()
        for i in range(inputs + 1):
            weights.append(weight_gen(inputs))

        hidden_nodes.append(weights)

    # initializing the weights for the output node
    output_node = list()
    for m in range(hidden + 1):
        output_node.append(weight_gen(hidden))

    # printing the node weights
    print('Nodes')
    for nodes in hidden_nodes:
        print(nodes)

    print()
    print(output_node)

    # starting the training
    for epoch in range(epochs):
        # looping through the training data
        for t in train:
            outs = list()
            # getting the results from forward pass for the hidden layer nodes
            for node in hidden_nodes:
                if tanh:
                    outs.append(output_tan(activation(node, dict_to_list(t))))
                else:
                    outs.append(output(activation(node, dict_to_list(t))))

            # beginning the backpropagation
            if tanh:
                final_out = output_tan(activation(output_node, outs))
                observed.append(output_tan(activation(output_node, outs)))
                delta_output = delta_out(dict_to_list(t)[-1], final_out, output_derivative_tan(final_out))

            else:
                final_out = output(activation(output_node, outs))
                observed.append(output(activation(output_node, outs)))
                delta_output = delta_out(dict_to_list(t)[-1], final_out, output_derivative(final_out))

            # saving the real value of index flood
            real.append(dict_to_list(t)[-1])
            deltas = list()

            # calculating the deltas for the hidden nodes
            for node in range(len(hidden_nodes)):
                if tanh:
                    deltas.append(delta_hidden(output_node[node], delta_output, output_derivative_tan(outs[node])))
                else:
                    deltas.append(delta_hidden(output_node[node], delta_output, output_derivative(outs[node])))

            # weight adjustment for the nodes
            for nodes in range(len(hidden_nodes)):
                hidden_nodes[nodes] = weight_adjustment(hidden_nodes[nodes], dict_to_list(t), deltas[nodes],
                                                        learning_parameter)
            output_node = weight_adjustment(output_node, outs, delta_output, learning_parameter)

        # calculating the MSE, RMSE and MSRE values
        summ = 0.0
        for n in range(len(train)):
            summ += math.pow((observed[n] - real[n]), 2)

        observed.clear()
        real.clear()
        error.append(summ / len(train))
        rmse_value.append(rmse(hidden_nodes, output_node, validation, tanh))
        msre_values.append(msre(hidden_nodes, output_node, testing, tanh))

    # plotting the errors
    plt.plot(error)
    plt.title('Error MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.savefig("MSE " + str(epochs) + " " + str(learning_parameter) + " " + str(hidden) + ' ' + str(tanh) + " .jpeg")
    plt.show()

    plt.plot(rmse_value)
    plt.title('Error RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.savefig("RMSE " + str(epochs) + " " + str(learning_parameter) + " " + str(hidden) + ' ' + str(tanh) + " .jpeg")
    plt.show()

    plt.plot(msre_values)
    plt.title('Error MSRE')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.savefig("MSRE " + str(epochs) + " " + str(learning_parameter) + " " + str(hidden) + ' ' + str(tanh) + " .jpeg")
    plt.show()

    # printing the last errors and weights
    print('Error Final: ' + str(error[-1]))
    print('RMSE Final: ' + str(rmse_value[-1]))
    print('MSRE Final Error: ' + str(msre_values[-1]))

    print("Output weights")
    print(output_node)
    print()

    for nodes in hidden_nodes:
        print(nodes)

    # returns the hidden layer weights and output node weights
    return hidden_nodes, output_node


# sums the weights and their inputs
def activation(weights, inputs):
    activated = weights[-1]

    for i in range(len(weights) - 1):
        activated = activated + (inputs[i] * weights[i])

    return activated


# outputs the result of the sigmoid function
def output(sum_weights):
    return 1.0 / (1.0 + math.exp(-sum_weights))


# outputs the result of the tanh function
def output_tan(sum_weights):
    return (math.exp(sum_weights) - math.exp(-sum_weights)) / (math.exp(sum_weights) + math.exp(-sum_weights))


# outputs the result of the inverse of sigmoid function
def output_derivative(inputted):
    return inputted * (1.0 - inputted)


# outputs the result of the inverse of the tanh function
def output_derivative_tan(inputted):
    return 1 - math.pow(inputted, 2)


# calculates the delta of the hidden nodes
def delta_hidden(weight, delta_outt, output_deriv):
    return weight * delta_outt * output_deriv


# calculates the delta of the output node
def delta_out(correct, outputt, output_deriv):
    return (correct - outputt) * output_deriv


# calculates the weight adjustment made into the weights
def weight_adjustment(weights, inputs, delta, learning_param):
    weights[-1] = weights[-1] + learning_param * delta

    for x in range(len(weights) - 1):
        weights[x] = weights[x] + learning_param * delta * inputs[x]

    return weights


# generates the weights used in the nodes
def weight_gen(n):
    return random.uniform((-2 / n), (2 / n))


# takes a dictionary and it returns a list
def dict_to_list(diction):
    return list(diction.values())


# distandardises the output of the MLP
def distandardise(max_value, min_value, value):
    return ((value - 0.1) / 0.8) * (max_value - min_value) + min_value


# calculates the RMSE value
def rmse(hidden, out_node, dataset, tanh):
    observed = list()
    real = list()
    # initializes the list that save the values

    # runs the front pass
    for t in dataset:
        outs = list()
        for node in hidden:
            if tanh:
                outs.append(output_tan(activation(node, dict_to_list(t))))
            else:
                outs.append(output(activation(node, dict_to_list(t))))

        # saves the corrrect and predicted values
        observed.append(output(activation(out_node, outs)))
        real.append(dict_to_list(t)[-1])

    # makes final calculation of RMSE
    summed = 0.0
    for i in range(len(observed)):
        summed += math.pow((observed[i] - real[i]), 2)
    return math.sqrt((summed / len(real)))


# calculates the MSRE value
def msre(hidden, out_node, dataset, tanh):
    observed = list()
    real = list()
    # initializes the list that save the values

    # runs the front pass
    for t in dataset:
        outs = list()
        for node in hidden:
            if tanh:
                outs.append(output_tan(activation(node, dict_to_list(t))))
            else:
                outs.append(output(activation(node, dict_to_list(t))))

        # saves the corrrect and predicted values
        observed.append(output(activation(out_node, outs)))
        real.append(dict_to_list(t)[-1])

    # makes final calculation of MRSE
    summed = 0.0
    for i in range(len(observed)):
        summed += math.pow(((observed[i] - real[i]) / real[i]), 2)
    return summed / len(real)


# Prints the plot of the modelled and predicted data
def print_and_plot_test_dataset(hidden, output_node, dataset, algorithm, tanh):
    file = open(algorithm + 'Ploted.csv', 'w+')
    file.write('Modelled,Real \n')
    observed = list()
    real = list()
    # initializes the list that save the values and opnes the csv file to write the data

    # runs the front pass
    for t in dataset:
        outs = list()
        for node in hidden:
            if tanh:
                outs.append(output_tan(activation(node, dict_to_list(t))))
            else:
                outs.append(output(activation(node, dict_to_list(t))))

        # saves the corrrect and predicted valuesin their respected list and CSV File
        observed.append(output(activation(output_node, outs)))
        file.write(str(output(activation(output_node, outs))) + ',' + str(dict_to_list(t)[-1]) + '\n')
        real.append(dict_to_list(t)[-1])

    file.close()
    # plotting the correct and predicted values
    plt.plot(observed, label='Modelled')
    plt.plot(real, label='Real')
    plt.title(algorithm)
    plt.ylabel('Standardised Values')
    plt.xlabel('Row of Data')
    plt.legend()
    plt.savefig(algorithm + ' data plotted.jpeg')
    plt.show()


if __name__ == '__main__':
    net = network_ini(3, 3, csv_reader.input_limited_training(), csv_reader.input_limited_validation(),
                      csv_reader.input_limited_validation(), 1500, 0.1, True)
    print_and_plot_test_dataset(net[0], net[-1], csv_reader.input_limited_validation(), 'Simple Backpropagation 1500 '
                                                                                        '3 3 using Test Tanh', True)
