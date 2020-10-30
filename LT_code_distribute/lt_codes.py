#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
#import argparse
import numpy as np
#import core
from core import *
from distributions import *
from encoder import encode
from decoder import decode

def blocks_read(matrix, blocks_n):
    """ 
       给matrix进行row方向分块，分块数目为blocks_n
    """
    blocks = np.vsplit(matrix, blocks_n)
    return blocks

def blocks_write(blocks):
    """ 对计算好的blocks进行组合得到最终的结果
    """
    blocks_n = len(blocks)
    result = np.vstack(blocks)
    return result

#########################################################
    
if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(description="Robust implementation of LT Codes encoding/decoding process.")
    parser.add_argument("filename", help="file path of the file to split in blocks")
    parser.add_argument("-r", "--redundancy", help="the wanted redundancy.", default=2.0, type=float)
    parser.add_argument("--systematic", help="ensure that the k first drops are exactaly the k first blocks (systematic LT Codes)", action="store_true")
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--x86", help="avoid using np.uint64 for x86-32bits systems", action="store_true")
    args = parser.parse_args()

    core.NUMPY_TYPE = np.uint32 if args.x86 else core.NUMPY_TYPE
    core.SYSTEMATIC = True if args.systematic else core.SYSTEMATIC 
    core.VERBOSE = True if args.verbose else core.VERBOSE    

    with open(args.filename, "rb") as file:

        print("Redundancy: {}".format(args.redundancy))
        print("Systematic: {}".format(core.SYSTEMATIC))

        filesize = os.path.getsize(args.filename)
        print("Filesize: {} bytes".format(filesize))

        # Splitting the file in blocks & compute drops
        file_blocks = blocks_read(file, filesize)
        file_blocks_n = len(file_blocks)
        drops_quantity = int(file_blocks_n * args.redundancy)

        print("Blocks: {}".format(file_blocks_n))
        print("Drops: {}\n".format(drops_quantity))

        # Generating symbols (or drops) from the blocks
        file_symbols = []
        for curr_symbol in encode(file_blocks, drops_quantity=drops_quantity):
            file_symbols.append(curr_symbol)

        # HERE: Simulating the loss of packets?

        # Recovering the blocks from symbols
        recovered_blocks, recovered_n = decode(file_symbols, blocks_quantity=file_blocks_n)
        
        if core.VERBOSE:
            print(recovered_blocks)
            print("------ Blocks :  \t-----------")
            print(file_blocks)

        if recovered_n != file_blocks_n:
            print("All blocks are not recovered, we cannot proceed the file writing")
            exit()

        splitted = args.filename.split(".")
        if len(splitted) > 1:
            filename_copy = "".join(splitted[:-1]) + "-copy." + splitted[-1] 
        else:
            filename_copy = args.filename + "-copy"

        # Write down the recovered blocks in a copy 
        with open(filename_copy, "wb") as file_copy:
            blocks_write(recovered_blocks, file_copy, filesize)

        print("Wrote {} bytes in {}".format(os.path.getsize(filename_copy), filename_copy))

        '''
    MatrixA = np.arange(0,10000,1)
    MatrixA = MatrixA.reshape(1000,10)
    print(MatrixA)
    #将矩阵MatrixA 分成10份
    matrix_blocks = blocks_read(MatrixA, 10)
    matrix_blocks_n = len(matrix_blocks)
    drops_quantity = int(matrix_blocks_n * 5) # redundancy=5
    # Generating symbols (or drops) from the blocks
    matrix_symbols = []
    for curr_symbol in encode(matrix_blocks, drops_quantity=drops_quantity):
        curr_symbol = Symbol(curr_symbol.index, curr_symbol.degree, curr_symbol.data*1.5) #对编码过后的符号进行右乘操作，测试成功
        matrix_symbols.append(curr_symbol)

    # Recovering the blocks from symbols
    recovered_blocks, recovered_n = decode(matrix_symbols, blocks_quantity=matrix_blocks_n)
    Matrix_result = blocks_write(recovered_blocks)
    print(Matrix_result)