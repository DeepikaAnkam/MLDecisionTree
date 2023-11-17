# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
# Anjum Chida (anjum.chida@utdallas.edu)
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz
import math

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values_data (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values_data in the vector z.
    """
    # values_data = np.unique(x)
    # indices = {}
    # for v in values_data:
    #     indices[v] = [i for i, val in enumerate(x) if val == v]
    # return indices
    # raise Exception('Function not yet implemented!')

    unique_values = np.unique(x)
    indices_dict = {}

    for unique_val in unique_values:
      indices = []
      for i in range(len(x)):
        if x[i] == unique_val:
          indices.append(i)
      indices_dict[unique_val] = indices
    return indices_dict
    raise Exception('Function not yet implemented!')



def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values_data (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # E = 0
    # values_data, counts = np.unique(y, return_counts=True)
    # for c in counts:
    #     E = E + (c/len(y) * math.log(c/len(y), 2))
    # return -E
    # raise Exception('Function not yet implemented!')

    entropy = 0
    unique_values, value_counts = np.unique(y, return_counts=True)
    
    for count in value_counts:
        p = count / len(y)
        entropy = entropy + (p * math.log(p, 2))
    return -entropy
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # MI = {}
    # values_data, counts = np.unique(x, return_counts=True)
    # Entropy_y = entropy(y)

    # for v in values_data:
    #     new_y1 = y[np.where(x == v)]
    #     new_y2 = y[np.where(x != v)]
    #     MI[v] = (Entropy_y - ((len(new_y1)/len(y)*entropy(new_y1)) + (len(new_y2)/len(y)*entropy(new_y2))))
    # return MI
    # raise Exception('Function not yet implemented!')

    mutual_info = {}

    unique_values, value_counts = np.unique(x, return_counts=True)
    entropy_y = entropy(y)

    for value in unique_values:
        subset_y1 = y[x == value]
        subset_y2 = y[x != value]
        mutual_info[value] = entropy_y - (
            (len(subset_y1) / len(y) * entropy(subset_y1)) +
            (len(subset_y2) / len(y) * entropy(subset_y2))
        )

    return mutual_info

    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values_data of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values_data a, b, c) and x2 (taking values_data d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values_data:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    # values_data = np.unique(y)

    # if len(values_data) == 1:
    #     return values_data[0]

    # if attribute_value_pairs == 0:
    #     values_data, counts = np.unique(y, return_counts=True)
    #     return values_data[np.argmax(counts)]

    # if depth == max_depth:
    #     values_data, counts = np.unique(y, return_counts=True)
    #     return values_data[np.argmax(counts)]

    # decision_tree = {}
    # gain_max_att = 0
    # max_att = 0
    # max_att_value = 0
    # for att, att_value in attribute_value_pairs:
    #     MI = mutual_information(x[:,att], y)
    #     gain_att_value = MI.get(att_value)
    #     if gain_att_value == None:
    #         continue

    #     max_gain_MI = gain_att_value
    #     max_value_MI = att_value
    #     if ((max_gain_MI > gain_max_att)):
    #         gain_max_att = max_gain_MI
    #         max_att = att
    #         max_att_value = max_value_MI

    # if((gain_max_att != 0) or (max_att != 0) or (max_att_value != 0)):
    #     attribute_value_pairs.remove((max_att, max_att_value))
    #     true_side = (max_att, max_att_value, True)
    #     false_side = (max_att, max_att_value, False)
    #     partition_x = partition(x[:,max_att])
    #     x_att_indices = partition_x[max_att_value]
    #     false_y = y[np.where(x[: ,max_att] != max_att_value)]
    #     false_x = x[np.where(x[: ,max_att] != max_att_value)]
    #     a = id3(x[x_att_indices], y[x_att_indices], attribute_value_pairs, depth+1, max_depth)
    #     b = id3(false_x, false_y, attribute_value_pairs, depth+1, max_depth)
    #     decision_tree = {true_side: a, false_side: b}
    # else:
    #     values_data, counts = np.unique(y, return_counts = True)
    #     return values_data[np.argmax(counts)]
    # return decision_tree
    
    
    unique_values = np.unique(y)

    # Termination condition 1: If all labels are the same, return that label
    if len(unique_values) == 1:
        return unique_values[0]

    # Termination condition 2: If no attribute-value pairs are left, return the majority label
    if attribute_value_pairs is None:
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    # Termination condition 3: If maximum depth is reached, return the majority label
    if depth == max_depth:
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    decision_tree = {}
    max_gain = 0
    best_attribute = None
    best_value = None

    for attribute, value in attribute_value_pairs:
        information_gain = mutual_information(x[:, attribute], y).get(value, None)

        if information_gain is None:
            continue

        if information_gain > max_gain:
            max_gain = information_gain
            best_attribute = attribute
            best_value = value

    if best_attribute is not None and best_value is not None:
        attribute_value_pairs.remove((best_attribute, best_value))
        true_branch = (best_attribute, best_value, True)
        false_branch = (best_attribute, best_value, False)

        partition_indices = partition(x[:, best_attribute])[best_value]
        false_y = y[x[:, best_attribute] != best_value]
        false_x = x[x[:, best_attribute] != best_value]

        true_subtree = id3(x[partition_indices], y[partition_indices], attribute_value_pairs, depth + 1, max_depth)
        false_subtree = id3(false_x, false_y, attribute_value_pairs, depth + 1, max_depth)

        decision_tree = {true_branch: true_subtree, false_branch: false_subtree}
    else:
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    return decision_tree

    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # for att_value_decision, child_tree in tree.items():
    #     att = att_value_decision[0]
    #     att_value = att_value_decision[1]
    #     decision = att_value_decision[2]

    #     if decision == (x[att] == att_value):
    #         if type(child_tree) is dict:
    #             label = predict_example(x, child_tree)
    #         else:
    #             label = child_tree

    #         return label
    
    for attribute_value_decision, sub_tree in tree.items():
        attribute, value, decision = attribute_value_decision

        if decision == (x[attribute] == value):
            if type(sub_tree) is dict:
                predicted_label = predict_example(x, sub_tree)
            else:
                predicted_label = sub_tree

            return predicted_label
    
    
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # count = 0
    # n = len(y_true)
    # for y_t, y_p in zip(y_true, y_pred):
    #     if y_t != y_p:
    #         count = count + 1
    # return (1/n)*count
    
    error_count = 0

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label != predicted_label:
            error_count += 1

    n = len(y_true)
    error_rate = (1 / n) * error_count

    return error_rate
    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    
    train_error1 = {}
    train_error2 = {}
    train_error3 = {}
    test_error1 = {}
    test_error2 = {}
    test_error3 = {}
    
    for depth in range(1,11):
        for dataset in range(1,4):
            training_set = '../monks_data/monks-' + str(dataset) + '.train'
            M = np.genfromtxt(training_set, missing_values=0, skip_header=0, delimiter=',', dtype=int)
            y_train = M[:, 0]
            x_train = M[:, 1:]
            
            test_set = '../monks_data/monks-' + str(dataset) + '.test'
            M = np.genfromtxt(test_set, missing_values=0, skip_header=0, delimiter=',', dtype=int)
            y_test = M[:, 0]
            x_test = M[:, 1:]
            
            attribute_value_pairs = [] 
            for att in range(len(x_train[0])):
                values_data = np.unique(x_train[:,att])
                for v in range(len(values_data)):
                    attribute_value_pairs.append((att,values_data[v]))
                    
            decision_tree = id3(x_train, y_train, attribute_value_pairs, max_depth=depth)
            
            # Pretty print it to console
            pretty_print(decision_tree)
            
            # Visualize the tree and save it as a PNG image
            dot_str = to_graphviz(decision_tree)
            render_dot_file(dot_str, './my_learned_tree'+str(dataset)+'_'+str(depth))
            
            y_pred = [predict_example(x, decision_tree) for x in x_test]
            tst_err = compute_error(y_test, y_pred)
            if dataset == 1: test_error1[depth] = tst_err 
            if dataset == 2: test_error2[depth] = tst_err
            if dataset == 3: test_error3[depth] = tst_err
            
            y_pred = [predict_example(x, decision_tree) for x in x_train]
            trn_err = compute_error(y_train, y_pred)
            if dataset == 1: train_error1[depth] = trn_err 
            if dataset == 2: train_error2[depth] = trn_err
            if dataset == 3: train_error3[depth] = trn_err
                
import matplotlib.pyplot as plt

for n in range(1, 4):
        plt.figure(figsize=(12, 4))
        plt.title("Monks-" + str(n) + "dataset", fontsize=14)
        if n == 1:
            plt.plot(train_error1.keys(), train_error1.values(), marker='o', linewidth=4, markersize=10)
            plt.plot(test_error1.keys(), test_error1.values(), marker='o', linewidth=4, markersize=10)
        if n == 2:
            plt.plot(train_error2.keys(), train_error2.values(), marker='o', linewidth=4, markersize=10)
            plt.plot(test_error2.keys(), test_error2.values(), marker='o', linewidth=4, markersize=10)
        if n == 3:
            plt.plot(train_error3.keys(), train_error3.values(), marker='o', linewidth=4, markersize=10)
            plt.plot(test_error3.keys(), test_error3.values(), marker='o', linewidth=4, markersize=10)
        plt.xlabel('depth', fontsize=14)
        plt.ylabel('error', fontsize=14)
        plt.legend(['Train error', 'Test error'], fontsize=14)
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        plt.show()