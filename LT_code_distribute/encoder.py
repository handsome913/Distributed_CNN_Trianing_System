from LT_code_distribute.core import *
from LT_code_distribute.distributions import *

def get_degrees_from(distribution_name, N, k):
    """ Returns the random degrees from a given distribution of probabilities.
    The degrees distribution must look like a Poisson distribution and the 
    degree of the first drop is 1 to ensure the start of decoding.
    """

    if distribution_name == "ideal":
        probabilities = ideal_distribution(N)
    elif distribution_name == "robust":
        probabilities = robust_distribution(N)
    else:
        probabilities = None
    
    population = list(range(0, N+1))
    return [1] + choices(population, probabilities, k=k-1)
   
def encode(blocks, drops_quantity):
    """ Iterative encoding - Encodes new symbols and yield them.
    Encoding one symbol is described as follow:

    1.  Randomly choose a degree according to the degree distribution, save it into "deg"
        Note: below we prefer to randomly choose all the degrees at once for our symbols.

    2.  Choose uniformly at random 'deg' distinct input blocs. 
        These blocs are also called "neighbors" in graph theory.
    
    3.  Compute the output symbol as the combination of the neighbors.
        In other means, we XOR the chosen blocs to produce the symbol.

    

    迭代编码-编码新的符号并产生它们。编码一个符号描述如下:
    1. 根据度分布随机选择一个度，保存到“deg”中注:下面我们更喜欢随机选择所有的度一次为我们的符号。
    2. 均匀地随机选择'deg'不同的输入块。这些块在图论中也被称为“邻居”。  
    3. 将输出符号计算为邻居的组合。换句话说，我们用选定的块来产生符号。
    """

    # Display statistics
    blocks_n = len(blocks)
    assert blocks_n <= drops_quantity, "Because of the unicity in the random neighbors, it is need to drop at least the same amount of blocks"

    print("Generating graph...")
    start_time = time.time()

    # Generate random indexes associated to random degrees, seeded with the symbol id
    random_degrees = get_degrees_from("robust", blocks_n, k=drops_quantity)

    print("Ready for encoding.", flush=True)

    for i in range(drops_quantity):
        
        # Get the random selection, generated precedently (for performance)
        selection_indexes, deg = generate_indexes(i, random_degrees[i], blocks_n)

        # Xor each selected array within each other gives the drop (or just take one block if there is only one selected)
        drop = blocks[selection_indexes[0]]
        for n in range(1, deg):
            drop = drop + blocks[selection_indexes[n]]
            # drop = np.bitwise_xor(drop, blocks[selection_indexes[n]])
            # drop = drop ^ blocks[selection_indexes[n]] # according to my tests, this has the same performance

        # Create symbol, then log the process
        symbol = Symbol(index=i, degree=deg, data=drop)
        
        if VERBOSE:
            symbol.log(blocks_n)

        log("Encoding", i, drops_quantity, start_time)

        yield symbol

    print("\n----- Correctly dropped {} symbols (packet size={})".format(drops_quantity, PACKET_SIZE))
