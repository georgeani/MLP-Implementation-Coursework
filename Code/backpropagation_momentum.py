import math
from backpropagation_algorithm import weight_gen, rmse, activation, delta_out, delta_hidden, \
    dict_to_list, output, output_derivative, output_tan, output_derivative_tan, msre, print_and_plot_test_dataset
import matplotlib.pyplot as plt
import csv_reader


# The libraries used by the algorithm as explained in the cw
# The main network initialization function, it is in charge of the forward pass and backwards pass
# the input names are what they denote: hidden = hidden layer nodes
# train is the training dataset and validation is the validation dataset
# tanh denotes whether or not to use the tanh activation formula
# a is the a value from momentum, denoting how much momentum is applied
def network_ini_momentum(inputs, hidden, train, validation, test, epochs, learning_parameter, tanh, a):
    # initializing the arrays used to keep the weights of each node as well as some error metrics
    hidden_nodes = list()
    curr_hidden_nodes = list()
    hid_diff = [[0 for i in range(inputs)] for x in range(hidden)]
    rmse_value = list()
    msre_values = list()
    observed = list()
    real = list()
    error = list()

    # initializing the weights for the hidden nodes and updating the array that will have the previous values
    for h in range(hidden):
        weights = list()
        for i in range(inputs + 1):
            weights.append(weight_gen(inputs))

        hidden_nodes.append(weights)

    curr_hidden_nodes = hidden_nodes
    output_node = list()
    for m in range(hidden + 1):
        output_node.append(weight_gen(hidden))

    curr_out_node = output_node
    out_dif = [0 for i in range(hidden)]

    # printing the node weights
    print('Nodes')
    for nodes in hidden_nodes:
        print(nodes)

    print()
    print(output_node)

    # starting the training
    for epoch in range(epochs):
        # looping through the training data
        for t in range(len(train)):
            outs = list()
            # getting the results from forward pass for the hidden layer nodes
            for node in hidden_nodes:
                if tanh:
                    outs.append(output_tan(activation(node, dict_to_list(train[t]))))
                else:
                    outs.append(output(activation(node, dict_to_list(train[t]))))

            # beginning the backpropagation
            if tanh:
                final_out = output_tan(activation(output_node, outs))
                observed.append(output_tan(activation(output_node, outs)))
                delta_output = delta_out(dict_to_list(train[t])[-1], final_out, output_derivative_tan(final_out))

            else:
                final_out = output(activation(output_node, outs))
                observed.append(output(activation(output_node, outs)))
                delta_output = delta_out(dict_to_list(train[t])[-1], final_out, output_derivative(final_out))

            # saving the real value of index flood
            real.append(dict_to_list(train[t])[-1])
            deltas = list()

            # calculating the deltas for the hidden nodes
            for node in range(len(hidden_nodes)):
                if tanh:
                    deltas.append(delta_hidden(output_node[node], delta_output, output_derivative_tan(outs[node])))
                else:
                    deltas.append(delta_hidden(output_node[node], delta_output, output_derivative(outs[node])))

            # weight adjustment for the nodes
            h_node = list()
            for nodes in range(len(hidden_nodes)):
                h_node.append(weight_adjustment_momentum(hidden_nodes[nodes], dict_to_list(train[t]), deltas[nodes],
                                                         learning_parameter, a, hid_diff[nodes]))
                hid_diff[nodes] = difference_weight(h_node[nodes], curr_hidden_nodes[nodes])
            hidden_nodes = h_node

            curr_hidden_nodes = hidden_nodes

            output_node = weight_adjustment_momentum(output_node, outs, delta_output, learning_parameter, a,
                                                     out_dif)
            out_dif = difference_weight(output_node, curr_out_node)

            curr_out_node = output_node

        # calculating the MSE, RMSE and MSRE values
        summ = 0.0
        for n in range(len(train)):
            summ += math.pow((observed[n] - real[n]), 2)

        observed.clear()
        real.clear()
        error.append(summ / len(train))
        rmse_value.append(rmse(hidden_nodes, output_node, validation, tanh))
        msre_values.append(msre(hidden_nodes, output_node, test, tanh))

    # plotting the errors
    plt.plot(error)
    plt.title('Error MSE Momentum')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.savefig(
        "MSE Momentum " + str(epochs) + " " + str(learning_parameter) +
        " " + str(hidden) + ' ' + str(a) + ' ' + str(tanh) + " .jpeg")
    plt.show()

    plt.plot(rmse_value)
    plt.title('Error RMSE Momentum')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.savefig(
        "RMSE Momentum" + str(epochs) + " " + str(learning_parameter) +
        " " + str(hidden) + ' ' + str(a) + ' ' + str(tanh) + " .jpeg")
    plt.show()

    plt.plot(msre_values)
    plt.title('Error MSRE')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.savefig(
        "MSRE Momentum " + str(epochs) + " " + str(learning_parameter) +
        " " + str(hidden) + ' ' + str(a) + ' ' + str(tanh) + " .jpeg")
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


# this function helps to calculate the difference between the new and the old weights
def difference_weight(new_weights, old_weights):
    dif = list()
    # initializing an array that holds the weight difference from both weights

    for i in range(len(new_weights)):
        dif.append(new_weights[i] - old_weights[i])

    return dif


# this function mimics weight adjustment function from the simple backpropagation algorithm, yet it takes into account
# the weight difference and a
def weight_adjustment_momentum(weights, inputs, delta, learning_param, a, difference_weights):
    out_weight = list()
    # initializing the list that will hold the new weights

    # calculating weight difference
    for x in range(len(weights) - 1):
        out_weight.append(weights[x] + learning_param * delta * inputs[x] + a * difference_weights[x])

    out_weight.append(weights[-1] + learning_param * delta + a * difference_weights[-1])

    # returning the new weights
    return out_weight


if __name__ == '__main__':
    net = network_ini_momentum(8, 8, csv_reader.standardisedTraining(), csv_reader.standardisedValidation(),
                               csv_reader.standardisedValidation(), 750, 0.1, False, 0.9)
    print_and_plot_test_dataset(net[0], net[-1], csv_reader.standardised_testing(),
                                'Momentum Backpropagation 750 8 8 using Test ',
                                False)
