from matplotlib import pyplot as plt

__all__ = ["plot_root"]

def plot_root(index, data):
    """
    :param index: index of the function
    :param data: dictionary of the methods and their data containing the errors and iterations
    :return: None
    
    Description: plot the absolute errors vs iterations for each method
    """
    
    # add another key for the iteration list
    for key in data.keys():
        data[key]['iterations'] = [i for i in range(1, len(data[key]['errors'])+1)]
         
    Figure = plt.figure()
    
    # plot the erros vs iterations for each method
    for key in data.keys():
        plt.plot(data[key]['iterations'], data[key]['errors'], label=key)
    
    plt.xlabel("Iterations")
    plt.ylabel("Absolute Errors")
    plt.title("Absolute Errors vs Iterations", )
    plt.legend(title="Root finding methods", loc="upper right",)
    
    
    # plt.savefig(f'root_finding_methods_{index}.png')
    plt.show()