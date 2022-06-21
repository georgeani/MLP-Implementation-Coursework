import math

import matplotlib.pyplot as plt

from backpropagation_algorithm import weight_gen, rmse, activation, delta_out, delta_hidden, \
    dict_to_list, output, output_derivative, output_tan, output_derivative_tan, msre, print_and_plot_test_dataset
from backpropagation_momentum import difference_weight, weight_adjustment_momentum

from backpropagation_annealing import output_annealing

import csv_reader


# The libraries used by the algorithm as explained in the cw
# The main network initialization function, it is in charge of the forward pass and backwards pass
# the input names are what they denote: hidden = hidden layer nodes
# train is the training dataset and validation is the validation dataset
# start_param and end_param are the starting and ending parameters for annealing
# tanh denotes whether or not to use the tanh activation formula
# a is the a value from momentum, denoting how much momentum is applied
def network_ini_momentum_annealing(inputs, hidden, train, validation, testing, epochs, start_param, end_param, tanh, a):
    # initializing the arrays used to keep the weights of each node as well as some error metrics
    # this also includes the arrays that will have the weight differences, as well as the ones that will
    # have the previous values
    hidden_nodes = list()
    curr_hidden_nodes = list()
    hid_diff = [[0 for i in range(inputs)] for x in range(hidden)]
    rmse_value = list()
    msre_values = list()
    learning_value = list()
    # file = open('res.csv', "w+")
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
    for epoch in range(1, epochs + 1):
        # changing learning parameter
        learning_parameter = output_annealing(start_param, end_param, epoch, epochs)
        # looping through the training data
        learning_value.append(learning_parameter)
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
        msre_values.append(msre(hidden_nodes, output_node, testing, tanh))

    # plotting the errors
    plt.plot(error)
    plt.title('Error MSE Momentum Annealing')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.savefig(
        "MSE Momentum Annealing " + str(epochs) + " " + str(hidden) + ' ' + str(start_param) + ' ' + str(end_param) +
        ' ' + str(a) + ' ' + str(tanh) + " .jpeg")
    plt.show()

    plt.plot(rmse_value)
    plt.title('Error RMSE Momentum Annealing')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.savefig(
        "RMSE Momentum Annealing " + str(epochs) + " " + str(hidden) + ' ' + str(start_param) + ' ' + str(end_param) +
        ' ' + str(a) + ' ' + str(tanh) + " .jpeg")
    plt.show()

    plt.plot(learning_value)
    plt.title('Learning Parameter Momentum Annealing')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Parameter')
    plt.savefig(
        "Learning Parameter Momentum Annealing " + str(epochs) + " " + str(hidden) + ' ' + str(start_param) + ' ' +
        str(end_param) + ' ' + str(a) + ' ' + str(tanh) + " .jpeg")
    plt.show()

    plt.plot(msre_values)
    plt.title('Error MSRE')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.savefig(
        "MSRE Anealing Momentum " + str(epochs) + " " + str(hidden) + ' ' + str(start_param) + ' ' + str(end_param) +
        ' ' + str(a) + ' ' + str(tanh) + " .jpeg")
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


if __name__ == '__main__':
    net = network_ini_momentum_annealing(8, 8, csv_reader.standardisedTraining(), csv_reader.standardisedValidation(),
                                         csv_reader.standardisedValidation(), 1500, 0.1, 0.01, False, 0.9)
    print_and_plot_test_dataset(net[0], net[-1], csv_reader.standardised_testing(),
                                'Annealing Momentum 1500 8 8 using Test ',
                                False)
