import sys
print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv), str(sys.argv) )
print("-1: " , sys.argv[1], type(sys.argv[1]))
arg2 = int(sys.argv[2])
print("-2: " , arg2, type(arg2))
print("\n")