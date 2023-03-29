from matplotlib import pyplot as plt

__all__ = ["plotRootFindingMethods"]

def plotRootFindingMethods(index, data):
    """
    :param index: index of the function
    :param data: dictionary of the methods and their data containing the errors and iterations
    :return: None
    
    Description: plot the absolute errors vs iterations for each method
    """
    
    ## add another key for the iteration list
    for key in data.keys():
        data[key]['iterations'] = [i for i in range(1, len(data[key]['errors'])+1)]
         

    Figure = plt.figure(figsize=(10, 10))
    Figure.set_facecolor("white")
    Figure.set_edgecolor("black")
    Figure.set_linewidth(2)
    Figure.set_frameon(True)
    Figure.set_dpi(100)
    
    ## plot the erros vs iterations for each method
    for key in data.keys():
        plt.plot(data[key]['iterations'], data[key]['errors'], color=data[key]['color'], label=key)
    
    plt.xlabel("Iterations", fontsize=15)
    plt.ylabel("Absolute Errors", fontsize=15)
    plt.title("Absolute Errors vs Iterations", fontsize=20)
    plt.legend(title="Root finding methods", loc="upper right", fontsize=20)
    
    
    plt.savefig(f'root_finding_methods_{index}.png')
    plt.show()