# Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.utils import register_keras_serializable
import numpy as np

# float point precision
tf.keras.backend.set_floatx('float64')

# Main building function
def sol_to_yann(solution, inputs):

    # Needed params
    num_inputs = inputs
    big_M = 500 #bounding variable within the relu nodes

    # Extract MPQP solution information
    E_list, F_list, A_list, B_list = get_matrices_list(solution)
    num_cons_ordered, total_cons = get_cons_ordered(F_list)
    num_regions = len(F_list)

    # Custom activation function for binary step
    tf.keras.utils.get_custom_objects().clear()
    @register_keras_serializable(package = "custom", name="binary_step")
    def binary_step_activation(x):
        return tf.where(x >= 0, 1.0, 0.0)

    # Define network input layer
    input_layer = Input(shape=(num_inputs,), name='input')

    # First layer with custom activation function
    # Checks which polytopic constraints are active
    l1 = Dense(total_cons, name='layer1')(input_layer)
    l1 = Lambda(binary_step_activation, output_shape=(total_cons,))(l1) 

    # Second layer, checks which regions are active
    l2 = Dense(num_regions, activation='relu', name='layer2')(l1)

    # Third layer, ensuring only 1 active region
    l3 = Dense(num_regions, activation='relu', use_bias=False, name='layer3')(l2)

    # Fourth layer, calculating the corresponding subfunction while supressing others
    concat = layers.Concatenate(name='concat')([l3, input_layer]) # This tells the model to reinject the original input
    l4 = Dense(2*num_regions*1, activation='relu', name='layer4')(concat)

    # Output layer, adding all of the subfunction outputs
    output_layer = Dense(1, activation='linear', use_bias=False, name='output')(l4)

    # Define model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    

    # Initialize weights with correct shapes
    layer1_weights = np.zeros((num_inputs,total_cons))
    layer1_biases = np.zeros((total_cons,))
    layer2_weights = np.zeros((total_cons,num_regions))
    layer2_biases = np.zeros(num_regions,)
    layer4_weights = np.zeros((num_regions+num_inputs,1*num_regions*2))
    layer4_biases = np.zeros((1*num_regions*2,))
    output_weights = np.zeros((1*num_regions*2,1))

    # layer 1 weights and biases
    cons_counter = 0
    for i in range(num_regions):
        # i is the region number
        E = E_list[i]
        F = F_list[i]
        j = num_cons_ordered[i]
        # j is the number of cons in the ith region
        layer1_weights[:,cons_counter:cons_counter+j] = E.T
        layer1_biases[cons_counter:cons_counter+j] = F.reshape(j,)
        cons_counter += j
    layer1_weights = -1 * layer1_weights

    # layer 2 weights and biases, collecting the binary outputs
    cons_counter = 0
    for i in range(num_regions):
        j = num_cons_ordered[i]
        layer2_weights[cons_counter:cons_counter+j,i] = np.ones((j,))
        layer2_biases[i] = np.array([(1-j)]).reshape(1,)
        cons_counter += j

    # layer 3 is fixed values
    layer3_weights = np.tril(-1*np.ones((num_regions,num_regions)) + 2*np.identity(num_regions))
    
    # layer 4 weights and biases, subfunction evaluation
    laws_counter = 0
    for i in range(num_regions):
        layer4_weights[i,2*laws_counter:2*(laws_counter+1)] = big_M
        layer4_weights[num_regions:num_regions+num_inputs,2*laws_counter:2*(laws_counter+1)] = \
        np.concatenate([A_list[i][0:1],-A_list[i][0:1]]).T
        layer4_biases[2*laws_counter:2*(laws_counter+1)] = np.concatenate([B_list[i][0:1],-B_list[i][0:1]]).reshape(2*1,)
        laws_counter += 1
    layer4_biases = -big_M + layer4_biases

    # Output weights, stacked identity matrices
    pos_out = np.identity(1)
    neg_out = -np.identity(1)
    repeat_out = np.zeros((2*1,1))
    for i in range(1):
        repeat_out[2*i:2*(i+1),:] = np.concatenate([[pos_out[i]],[neg_out[i]]])
    output_weights = np.tile(repeat_out,(num_regions,1))
    
    # Assignt the weights and biases
    set_layer_weights(model, 'layer1', layer1_weights, layer1_biases)
    set_layer_weights(model, 'layer2', layer2_weights, layer2_biases)
    set_layer_weights(model, 'layer3', layer3_weights)  # No bias
    set_layer_weights(model, 'layer4', layer4_weights, layer4_biases)
    set_layer_weights(model, 'output', output_weights)  # No bias

    return model

# Function to manually set weights and biases
def set_layer_weights(model, layer_name, weights, biases=None):
    layer = model.get_layer(name=layer_name)
    
    if biases is not None:
        layer.set_weights([np.array(weights), np.array(biases)])
    else:
        layer.set_weights([np.array(weights)])

# Find total number of regions
def get_num_regions(F_list):
    num_regions = len(F_list)
    return num_regions

# Make a list of E and F matrix arrays for each region
def get_matrices_list(solution):
    sol = solution
    index_max = len(solution.critical_regions)
    E_list = []
    f_list = []
    A_list = []
    b_list = []
    for i in range(index_max):
        E_list.append(sol.critical_regions[i].E)
        f_list.append(sol.critical_regions[i].f)
        A_list.append(sol.critical_regions[i].A)
        b_list.append(sol.critical_regions[i].b)
    return E_list, f_list, A_list, b_list

# Find the total number of constraints and make an ordered list for #/region
def get_cons_ordered(F_list):
    total_cons = 0
    num_cons_ordered = []
    for _, F_set in enumerate(F_list):
        temp = len(F_set)
        num_cons_ordered.append(temp)
        total_cons += len(F_set)
    return num_cons_ordered, total_cons