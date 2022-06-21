import math
from backpropagation_algorithm import weight_gen, weight_adjustment, rmse, activation, delta_out, delta_hidden, \
    dict_to_list, output, output_derivative, output_tan, output_derivative_tan, msre, print_and_plot_test_dataset
import matplotlib.pyplot as plt
import csv_reader


# The libraries used by the algorithm as explained in the cw
# The main network initialization function, it is in charge of the forward pass and backwards pass
# the input names are what they denote: hidden = hidden layer nodes
# train is the training dataset and validation is the validation dataset
# tanh denotes whether or not to use the tanh activation formula
# upper and lower limit denote the limits when bold driver will change the learning parameter
# activation cycle denotes how often bold driver is activated
def network_ini_bold_driver(inputs, hidden, train, validation, test, epochs, learning_parameter, tanh, upper_limit,
                            low_limit, activation_cycle):
    # initializing the arrays used to keep the weights of each node as well as some error metrics
    hidden_nodes = list()
    rmse_value = list()
    msre_values = list()
    leaning = list()
    observed = list()
    real = list()
    error = list()
    previous_hidden = list()
    previous_out = list()
    previous_error = 0.0

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
        # saves the learning parameter to check whether or not bold driver has been applied
        leaning.append(learning_parameter)
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

            for nodes in range(len(hidden_nodes)):
                hidden_nodes[nodes] = weight_adjustment(hidden_nodes[nodes], dict_to_list(train[t]), deltas[nodes],
                                                        learning_parameter)
            output_node = weight_adjustment(output_node, outs, delta_output, learning_parameter)

        summ = 0.0
        for n in range(len(train)):
            summ += math.pow((observed[n] - real[n]), 2)

        # calculating the MSE, RMSE and MSRE values
        observed.clear()
        real.clear()
        error.append(summ / len(train))
        rmse_value.append(rmse(hidden_nodes, output_node, validation, tanh))
        msre_values.append(msre(hidden_nodes, output_node, test, tanh))

        # the bold driver algorithm
        # it checks whether or not the correct epoch has come
        if epoch % activation_cycle == 0 and epoch > 0:
            # updating the learning parameter
            learn = modify_learning_param(learning_parameter, previous_error, error[-1], low_limit, upper_limit)
            # if learning parameter is different
            # we change the learning parameter an revert to the previously stored weights
            # we also reload the epoch
            if learning_parameter != learn:
                learning_parameter = learn
                previous_error = error[-1]
                hidden_nodes = previous_hidden
                output_node = previous_out
                epoch = epoch - 1
            else:
                # in case the learning parameter is the same we upgrade the weights we use
                previous_error = error[-1]
                previous_hidden = hidden_nodes
                previous_out = output_node
        elif epoch == 0:
            # in the 1st epoch we update the arrays and variables used by the algorithm
            print('first pass')
            previous_error = error[-1]
            previous_hidden = hidden_nodes
            previous_out = output_node
        leaning.append(learning_parameter)

    # plotting the errors
    plt.plot(error)
    plt.title('Error MSE')
    plt.xlabel('Epochs')
    plt.savefig("MSE Bold " + str(epochs) + " " + str(hidden) + " " + str(tanh) + " .jpeg")
    plt.show()

    plt.plot(rmse_value)
    plt.title('Error RMSE')
    plt.xlabel('Epochs')
    plt.savefig("RMSE Bold " + str(epochs) + " " + str(hidden) + " " + str(tanh) + " .jpeg")
    plt.show()

    plt.plot(leaning)
    plt.title('Learning Parameter Change')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Parameter')
    plt.savefig("Learning Parameter Bold " + str(epochs) + " " + str(hidden) + " " + str(tanh) + " .jpeg")
    plt.show()

    plt.plot(msre_values)
    plt.title('Error MSRE')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.savefig("MSRE Bold " + str(epochs) + " " + str(hidden) + " " + str(tanh) + " .jpeg")
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


# this function is used in order to modify the learning parameter when bold driver runs
def modify_learning_param(learning_param, previous_error, current_error, low_limit, upper_limit):
    learn = learning_param
    # calculate error deviation
    dif = current_error / previous_error
    print('Diference: ' + str(dif))
    # check error dviation
    if dif >= upper_limit:
        learn = learning_param * 0.7
    elif dif <= low_limit:
        learn = learning_param * 1.05

    # check whether or not the learning parameter is not too large or too small
    if learn <= 0.01 or learn >= 0.5:
        return learning_param
    else:
        return learn


if __name__ == '__main__':
    net = network_ini_bold_driver(8, 8, csv_reader.standardisedTraining(), csv_reader.standardisedValidation(),
                                  csv_reader.standardisedValidation(), 1500, 0.1, False, 1.1, 0.95, 200)
    print_and_plot_test_dataset(net[0], net[-1], csv_reader.standardised_testing(),
                                'Bold Backpropagation 1500 8 8 using Test ',
                                False)
