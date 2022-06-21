import math
from backpropagation_algorithm import weight_gen, weight_adjustment, rmse, activation, delta_out, delta_hidden, \
    dict_to_list, output, output_derivative, output_tan, output_derivative_tan, msre, print_and_plot_test_dataset
import matplotlib.pyplot as plt
import csv_reader


# The libraries used by the algorithm as explained in the cw
# The main network initialization function, it is in charge of the forward pass and backwards pass
# tanh denotes whether or not to use the tanh activation formula
# the input names are what they denote: hidden = hidden layer nodes
# train is the training dataset and validation is the validation dataset
# start_param and end_param are the starting and ending parameters for annealing
def network_ini_annealing(inputs, hidden, train, validation, epochs, start_param, end_param, tanh):
    # initializing the arrays used to keep the weights of each node as well as some error metrics
    hidden_nodes = list()
    rmse_value = list()
    msre_values = list()
    leaning = list()
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
    for epoch in range(1, epochs + 1):
        # changing learning parameter
        learning_parameter = output_annealing(end_param, start_param, epochs, epoch)
        # looping through the training data
        leaning.append(learning_parameter)
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
        msre_values.append(msre(hidden_nodes, output_node, validation, tanh))

    # plotting the errors
    plt.plot(error)
    plt.title('Error MSE')
    plt.xlabel('Epochs')
    plt.savefig("MSE Annealing " + str(epochs) + " " + str(hidden) + " " + str(tanh) + " .jpeg")
    plt.show()

    plt.plot(rmse_value)
    plt.title('Error RMSE')
    plt.xlabel('Epochs')
    plt.savefig(
        "RMSE Annealing " + str(epochs) + " " + str(hidden) + ' ' + str(start_param) + ' ' + str(end_param) +
        " " + str(tanh) + " .jpeg")
    plt.show()

    plt.plot(leaning)
    plt.title('Learning Parameter Change')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Parameter')
    plt.savefig(
        "Learning Parameter Annealing " + str(epochs) + " " + str(hidden) + ' ' + str(start_param) + ' ' +
        str(end_param) + " " + str(tanh) + " .jpeg")
    plt.show()

    plt.plot(msre_values)
    plt.title('Error MSRE')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.savefig(
        "MSRE Annealing " + str(epochs) + " " + str(hidden) + ' ' + str(start_param) + ' ' +
        str(end_param) + " " + str(tanh) + " .jpeg")
    plt.show()

    # printing the last errors and weights
    print('Error Final: ' + str(error[-1]))
    print('RMSE Final: ' + str(rmse_value[-1]))
    print('MSRE Final: ' + str(msre_values[-1]))

    print("Output weights")
    print(output_node)
    print()

    for nodes in hidden_nodes:
        print(nodes)

    # returns the hidden layer weights and output node weights
    return hidden_nodes, output_node


# this function is used in order to calculate the learning parameter
def output_annealing(end_param, start_param, max_epochs, cur_epoch):
    return end_param + (start_param - end_param) * (
            1.0 - (1.0 / (1.0 + math.exp(10.0 - ((20.0 * cur_epoch) / max_epochs)))))


if __name__ == '__main__':
    net = network_ini_annealing(8, 8, csv_reader.standardisedTraining(), csv_reader.standardisedValidation(), 1000, 0.1,
                                0.01,
                                False)
    print_and_plot_test_dataset(net[0], net[-1], csv_reader.standardised_testing(), 'Annealing 1000 real Testing', False)
