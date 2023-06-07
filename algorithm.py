import numpy as np  
import random
import math 
import csv
import operator
import timeit
import matplotlib.pyplot as plt

#Global variables started with values
EPSILON = 0.5
DELTA = 0.1
K=100000
MAX_SIZE = 15
MIN_SIZE = 10
HADAMARD = True


###########################################################
#algorithm 3 from page(13) as it's written
#Computing heavy pairs of a matrix

#Input X(matrix nXr) and K>1(enum)
#Output matrix H (number_of_heavy_pairs X 3) each row = {i , j , C˜ij} (first row, second row , estimated score)
def ComputingHeavyPairs(X,K_tag):
    n = len(X)
    norms = np.zeros((n,2))
    print("Calculating vectors norms")
    #calc and save normes index and value
    for i in range(n): 
        norms[i] = [i,np.linalg.norm(X[i])] 
    #sort normes according to value from low to high
    norms = sorted(norms, key = operator.itemgetter(1))
    #calculating X transpose * X
    XTX = np.matmul(X.transpose(),X)
    norm = 0
    #calculating normal squared for XTX ||XTX||^2
    for row in XTX: 
        for j in row:
            norm += pow(j,2)
    print(norm)
    print("norm/K_tag")
    print(norm/K_tag)
    print(len(X))
    print(len(X[0]))
    print(np.matrix(norms))
    z1 = n-1
    z2 = 0
    H = []
    while z2 <= z1:
        #||X1||^2  * ||X2||^2
        while (pow(norms[z1][1],2)*pow(norms[z2][1],2)<(norm/K_tag)):
            z2 += 1
            if z2>z1:
                return H
        j = z2      
        i = z1  
        #for j in range(z2,z1):
        while j <= i:
            #first value in norms [0] is index in X
            #np.multiply: multiplies the each value in the vector with the respective one in the other one
            #<Xi,Xj>^2
            inner_product = np.multiply(X[int(norms[i][0])],X[int(norms[j][0])]) 
            c = 0
            for x in inner_product:
                c += x
            c_pow_2 = pow(c,2)
            #check if it's a heavy pair 
            if c_pow_2 >= norm/K_tag:
                #{(i,j),c} in each column of size 3
                H.append([int(norms[i][0]),int(norms[j][0]),c]) 
            j += 1
            z1 -= 1
    return H   



##############################this part will be done prior to the run#############
#calculating HD matrix for SRHT will be run prior to the algorithm due to it not being related to the input
#be careful when running as matrix size rises exponentially and for 2^15 matrix will need 11GB of ram to calculate
#and 11GB to store

#Output transformations with sizes (nXn) n=[2^5 , 2^14]
def calculating_RHT():

    #creating D
    #diagonal matrix with random values of +/- 1 in the diagonal places
    for i in range(5,MAX_SIZE):
        n=pow(2,i)
        print("Creating diagonal matrix D")
        D = np.zeros((n,n))
        for i in range(n):
            D[i][i] = random.choice([-1,1])
        print("Creating hadamard matrix of size: ", n,"X",n)    
        H = hadamard(n,n)
        print("Multiplying Hadamrd transform with D to get a RHT")
        #replaced matmul with another function since D is a diagonal matrix and this way we can reduce complexity
        HD = diagonal_multiplication(H,D)
        HD.dump('HD{0}.dat'.format(n))

###########################################################


#Since one of the matrix is diagonal this multiplicaiton is O(n^2)
#we multiply just only one value from the row in H with the respective value in D

#Input: matrix mat(nXn), diagonal matrix diag(nXn)
#Output: matrix (nXn) the multiplication result 
def diagonal_multiplication(mat,diag):
    n = len(mat)
    result = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            result [i][j] = mat[i][j]*diag[j][j]
    return result



#The Subsampled Randomized Hadamard Transform (SRHT)

#Input: the size of matrix n=number of rows, d=number of columns
#output: matrix of size (rXn) 
def SRHT(n,d):
    #loading the randomized hadamard matrix calculated by the RHT function
    print("loading hadamard transform")
    HD = np.load('HD{0}.dat'.format(n),allow_pickle=True)
    print("Epsilon : ",EPSILON)
    #calculating r (number of rows for FJLT) according to lemma 6
    r = math.trunc(((d * math.log2(n))/pow(EPSILON,2))*(math.log2((d * math.log2(n))/pow(EPSILON,2))))

    #And sampling r rows from the RHT matrix randomly
    #incase the Epsilon is too small for the number of rows r might be bigger than n (because of the previus equasion)
    #in that case the algorithm doesn't subsample
    print("Sampling " ,r ,"rows out of ",n) 

    if(r < n): #condition since for small n the previous equasion returns big r

        # SRHT = np.zeros((r,n))
        # for i in range(r):
        #     row=random.randrange(n)
        #     for j in range(0,n):
        #         SRHT[i][j] = HD[row][j]

        #this code does the commented above 
        start = timeit.default_timer()
        SRHT = HD[np.random.choice(HD.shape[0], r, replace=False),:]
        stop = timeit.default_timer()
        print("sampling time: ",stop-start)
        return SRHT
    return HD #if r > n

    

#recursive function to build hadamard matrix not good recursion get too deep for big numbers
# def Hadamard(H,top,left,size,sign,n):
#     #print (size)
#     print(size)
#     if(size == 1):
#         top = math.trunc(top)
#         left = math.trunc(left)
#         H[top][left]=sign/(math.sqrt(n)) #normalization
#     else:
#         Hadamard(H,top,left,size/2,sign,n)
#         Hadamard(H,top+size/2,left,size/2,sign,n)
#         Hadamard(H,top,left+size/2,size/2,sign,n)
#         Hadamard(H,top+size/2,left+size/2,size/2,(-1)*sign,n) 
#     return H


#calculating hadamard trasformaion matrix recursivly assuming that n is a power of 2

#Input: size of square matrix: size = n = (2^x)
#Output: hadamard matrix(nXn)
def hadamard(size,n):
    if size == 1:
        #we normalize the matrix 
        return np.array([1/(math.sqrt(n))])
    elif size == 2:
        #we normalize the matrix 
        return np.array([[1/(math.sqrt(n)),1/(math.sqrt(n))],[1/(math.sqrt(n)),-1/(math.sqrt(n))]])
    else:
        a = hadamard(size // 2,n)
        return np.concatenate(
            (np.concatenate((a, a), axis=1),
            np.concatenate((a, -a), axis=1)),
            axis=0
        )

#calculating The Fast Johnson-Lindenstrauss Transform (FJLT) for n^2 vectors
#using the r_2 variable given in the algorithm according to lemma 4 we can calculate epsilon-JLT 
# with probablity < 1-Delta 

#Input: n=num of rows, d=num of columns
#Output: matrix PI_2 (dXr_2)  r_2=(epsilon^(-2) * ln(n))
def FJLT(n,d):
    r_2 = math.floor(pow(EPSILON,-2)*math.log2(n))
    PI_2 = np.zeros((r_2,d))
    num = math.sqrt(3/r_2)
    print("num")
    print(num)
    #choosing for each element in the matrix:
    # (3/r)^(1/2) with probability 1/6
    #-(3/r)^(1/2) with probability 1/6
    #0 with probability 2/3
    for row in range(r_2):
        for col in range(d):
            PI_2[row][col]=np.random.choice([num, -num, 0],1,p=[1/6,1/6,2/3])

    PI_2 = PI_2.transpose()
    return PI_2






##################################################################
#algorithm 1 page 11
#Approximating the (diagonal) statistical leverage scores ℓi

#Input A(matrix nXd) and Epsilon(enum (0 , 1/2])
#Output array(n) statistical scores
#in addition we Output matrix OMEGA(nXr_2) to be used in algorithm 2
def CalculatingStatisticalLeverageScores(df):
    ###########Step 1########### :
    n = len(df) #number of rows
    d = len(df[0]) #number of cols
    PI_1 = SRHT(n,d)
    #print(PI_1)


    ###########Step 2###########:
    print("Multiplying PI with the data frame")
    PI_A = np.matmul(PI_1,df) 
    print("calculating SVD")
    U, S, V = np.linalg.svd(PI_A)
    #inversing
    S = 1.0 / S
    #turnin s into diagonal matrix
    inv_s = np.zeros((len(S),len(S))) 
    for i in range(len(S)):
        inv_s[i][i] = S[i]

    trans_v=V.transpose()
    print("Multiplying V with invers S")
    inv_r=np.matmul(trans_v,inv_s)

    ###########Step 3###########:
    #Calculating The Fast Johnson-Lindenstrauss Transform (FJLT) for n^2 vectors

    print("Calculate Fast Johnson-Lindenstrauss Transform")
    PI_2 = FJLT(pow(n,2),d)
    #print(np.matrix(PI_2))

    ###########Step 4###########:

    #OMEGA = (AR^(-1))*PI
    print("Constructin OMEGA")
    AR = np.matmul(df,inv_r)

    #normalizing AR rows
    l2norms = np.sqrt((AR * AR).sum(axis=1))
    for i in range(len(AR)):
        AR[i] = AR[i]/l2norms[i]

    OMEGA = np.matmul(AR,PI_2) #nxr_2
    print(OMEGA)

    ###########Step 5###########:

    #the statistical scores are the squared normals of OMEGA rows
    statistical_scores = np.zeros(n) 
    for i in range(n):
        statistical_scores[i] = pow(np.linalg.norm(OMEGA[i]),2)
    return statistical_scores,OMEGA





##############################################################


#Approximating the large (off-diagonal) cross-leverage scores cij.
#algorithm 2:

#Input: df(nXd)
#Output matrix H (number_of_heavy_pairs X 3) each row = {i , j , C˜ij} (first row, second row , estimated score)
def Cross_leverage_scores(df):
    d=len(df[0])
    #calculating OMEGA using algorithm 1
    statistical_leverage_scores,OMEGA = CalculatingStatisticalLeverageScores(df)
    print("Calculating K':")
    #calculating K' accordint to step 2 (κ′ = κ(1 + 30dε)) 
    K_tag = math.floor(K * (1 + (30*d*EPSILON)))
    print(K_tag)
    #finding heavy pairs according to algorithm 3
    H = ComputingHeavyPairs(OMEGA, K_tag)
    print("-------- cross-leverage scores : --------")
    for i in range(len(H)):
        print("(",H[i][0],",",H[i][1],")"," : ",H[i][2])
    print("number of heavy pairs: ", len(H))
    return H

##############################################################


################################TESTS FOR ALGORITHM 1################################


#this test checks the differnce in values between the actual leverage scores and the 
#statistical leverage score calculated by our algorithm for different sizes of matrices
#to show the effect the number of rows has on how correct the values recieved are
#it works by summing the difference between each statistical leverage score and the curresponding 
#leverage scores and deviding by the number of scores at the end
def calculate_average_statistical_score_per_size():
    num=0
    diffs = np.zeros(5)
    times = np.zeros(5)
    rows = np.zeros(5)

    #we will try the algorithm for sizes [ 2^10 , 2^14 ]
    #sizes need to be to the power of 2
    for i in range(MIN_SIZE,MAX_SIZE):
        with open('magic04.data', 'rt') as file:
            df = [list(map(float, row[:-1])) for row in csv.reader(file,delimiter=",")]
        df=np.delete(df , slice(pow(2,i),19020) , axis = 0)
            
        print("Calculating leverage scores")
        n=len(df)
        
        #calculating the actual U matrix using svd function from numpy library
        U, S, V = np.linalg.svd(df)
        leverage_scores = np.zeros(len(df))
        curr = 0
        for row in U: 
            normalised_row = pow(np.linalg.norm(row),2)
            leverage_scores[curr] = normalised_row
            curr +=1
        
        #starting timer for algorithm 1
        start = timeit.default_timer()

        #calculating statistical leverage scores and OMEGA matrix using algorithm 1
        statistical_leverage_scores,OMEGA = CalculatingStatisticalLeverageScores(df)
        stop = timeit.default_timer()
        diff = 0

        #calculating the sum of the differences between the statistical leverage scores and the 
        #corresponding leverage scores
        for j in range(len(statistical_leverage_scores)):
            diff += abs(abs(statistical_leverage_scores[j])-abs(leverage_scores[j]))


        #deviding the sum of diffs by the number of scores to get the average diff    
        diffs[i-MIN_SIZE] = diff/(len(statistical_leverage_scores))

        #calculating runtime
        times[num] = stop - start
        rows[num] = pow(2,i)
        num += 1
        print('Time: ', stop - start)  
        print("averages : ",diffs)
        print("times : ", times)
    
    #plotting graphs
    plt.plot( rows,times) 
    plt.xlabel('rows')
    plt.ylabel('time')
    plt.title('number of rows to run time graph')
    plt.show()

    plt.plot(rows, diffs) 
    plt.xlabel('rows')
    plt.ylabel('leverage scores')
    plt.title('number of rows to average of leverage scores')
    plt.show()




#this test checks the differnce in values between the actual leverage scores and the 
#statistical leverage score calculated by our algorithm for different epsilon values
#to show the effect the value of epsilon has on how correct the values recieved are
#it works by summing the difference between each statistical leverage score and the curresponding 
#leverage scores and deviding by the number of scores at the end
def calculate_average_statistical_score_per_epsilon(A):
    #calculating leverage scores
    print("Calculating leverage scores")
    n=len(A)
    U, S, V = np.linalg.svd(A)
    leverage_scores = np.zeros(len(A))
    curr = 0
    
    #calculate the actual leverage scores
    for row in U: 
        normalised_row = pow(np.linalg.norm(row),2) #changed
        leverage_scores[curr] = normalised_row
        curr += 1

    diffs = np.zeros(5)
    Epsilons = np.zeros(5)
    successes = np.zeros(5)
    global EPSILON


    #we will run the test on different epsilon values
    #this test starts with epsilon = 0.5 and subtracts 0.05 each time for 5 runs
    for i in range(5):
        print("Epsilon: ",EPSILON)

        #get the statistical leverage scores by running algorithm 1 from the paper
        statistical_leverage_scores,OMEGA = CalculatingStatisticalLeverageScores(A)
        diff = 0
        success = 0

        #calculate the diff between each of the statistical leverage scores and the actual leverage score
        #didn't use the square distance because this will give us bigger diffs

        for j in range(len(statistical_leverage_scores)):
            diff += abs(statistical_leverage_scores[j]-leverage_scores[j])
            #print(EPSILON ,"  *  ",leverage_scores[j])
            #print((abs((abs(statistical_leverage_scores[j])-abs(leverage_scores[j]))) ,"  <=  ", abs(EPSILON*leverage_scores[j])))
            if(abs((abs(statistical_leverage_scores[j])-abs(leverage_scores[j]))) <= abs(EPSILON*leverage_scores[j])):
                success +=1

        #devide the diffs sum by the number of scores to get an average
        diffs[i] = diff/(len(statistical_leverage_scores))

        #calculating the percentage of statistical scores 
        successes[i] = success/(len(statistical_leverage_scores))

        print("diff = ", diffs)
        print("success = ",successes)
        Epsilons[i] = EPSILON
        EPSILON -= 0.05

    #plotting graphs
    plt.plot(Epsilons, diffs) 
    plt.ylabel('leverage scores')
    plt.xlabel('Epsilon')
    plt.title('Epsilon to the average of leverage scores')
    plt.show()

    plt.plot(Epsilons, successes) 
    plt.ylabel('success rate')
    plt.xlabel('Epsilon')
    plt.title('Epsilon to the success rate')
    plt.show()


################################TESTS FOR ALGORITHM 2################################
#this test will check if theorem 3 is true for algorithm 2 which needs two conditions to materialize
#for the first condition we check if the second part is true for the output
#Condition 1: If (C^2)ij ≥ d/κ+12εℓiℓj, then (i, j) is returned; if (i, j) is returned, then (C^2)ij ≥ d/κ−30εℓiℓj.
#Condition 2: For all pairs (i, j) that are returned, (C˜^2)ij −30εℓiℓj ≤ (C^2)ij ≤ (C˜^2)ij +12εℓiℓj.


def calculate_heavy_pairs_per_epsilon(A):
    #calculating leverage scores
    print("Calculating leverage scores")
    U, S, V = np.linalg.svd(A)
    d = len(A[0])
    results = np.zeros(5)
    Epsilons = np.zeros(5)
    global EPSILON
    EPSILON = 0.5
    number_of_heavy_pairs = np.zeros(5)
    #trying different Epsilon each time reducing Epsilong by 0.05
    for i in range(5):
        print("Epsilon: ",EPSILON)
        heavy_cross_leverage_scores = Cross_leverage_scores(A)
        success = 0
        number_of_heavy_pairs[i] = len(heavy_cross_leverage_scores)
        if len(heavy_cross_leverage_scores)>0:
            C = 0
            for j in range(len(heavy_cross_leverage_scores)):
                #get the two vectors from U
                U1 = U[heavy_cross_leverage_scores[j][0]]
                U2 = U[heavy_cross_leverage_scores[j][1]]

                #calculate cross leverage score: C(ij)=<U(i),U(j)>
                #multiply multiplies each value from one vector with the corresponding one 
                #from the other vector
                res = np.multiply(U1,U2)
                for num in res:
                    C += num
                C_2 = pow(C,2)
                
                #squared normals for Ui and Uj
                Li=pow(np.linalg.norm(U1),2)
                Lj=pow(np.linalg.norm(U2),2)

                #get the statistical cross leverage scores for the two vectors (output of algorithm 3)
                C_tag = heavy_cross_leverage_scores[j][2]
                C_tag_2 = pow(C_tag,2)

                #check if Theorem 3 succeeds
                #for that to happen these if statements need to be true with propability of at least 0.8
                if ((C_2 <= (C_tag_2 + 12 * EPSILON * Li * Lj)) and (C_2 >= (C_tag_2 - 30 * EPSILON * Li * Lj))):
                    if(C_2 >= ((d/K) - (30 * EPSILON * Li * Lj))):
                        #print("C^2 = ",C_2)
                        success += 1

            #calculate the success percentage that teorem 2 is true for them 
            result = success / len(heavy_cross_leverage_scores)
            results[i] = result
            print("success rate: ", result)
        else:
            print("no heavy pairs found")
            results[i] = 0
        Epsilons[i] = EPSILON
        EPSILON -= 0.025

    #plotting graphs
    plt.plot(Epsilons, results) 
    plt.ylabel('Success rate')
    plt.xlabel('Epsilon')
    plt.title('the success rate for heavy pairs for each epsilon')
    plt.show()
    
    plt.plot(Epsilons, number_of_heavy_pairs) 
    plt.ylabel('Number of heavy pairs')
    plt.xlabel('Epsilon')
    plt.title('Number of heavy pairs to Epsilon')
    plt.show()



################################END OF TESTS################################

#The main is algorithm 2 from the Paper
def main():

    print("START")
    print("Input is assumed to be power of 2")

    ########################## hadamard ##########################
    #if you didn't caculate the RHT matrix ( the hadamard transform multyplied by a random diagonal matrix)
    #you need to run the calculating_RHT() function
    # which will calculate the matrices for sizes [2^10 , 2^14]
    if(HADAMARD == False):
        calculating_RHT()


    ########################## data sets ##########################
    #uncomment the data set you want to use

    # with open('seeds_dataset.txt', 'rt') as file:
    #     df = [list(map(float, row)) for row in csv.reader(file,delimiter="\t")]
    # df=np.delete(df , slice(128,210) , axis = 0)

    with open('avila-tr.txt', 'rt') as file:
        A = [list(map(float, row[:-1])) for row in csv.reader(file,delimiter=",")]
    A=np.delete(A , slice(8192,10430) , axis = 0)

    with open('magic04.data', 'rt') as file:
        df = [list(map(float, row[:-1])) for row in csv.reader(file,delimiter=",")]
    df=np.delete(df , slice(16384,19020) , axis = 0)


    ########################## the algorithms ##########################
    #uncomment this section to run the algorithms 

    #first algorithm (needed for the second one)
    statistical_leverage_scores,OMEGA = CalculatingStatisticalLeverageScores(df)
    print("-------- statistical-leverage scores : --------")
    print(statistical_leverage_scores)
    print("number of statistical leverage scores: ", len(statistical_leverage_scores))
    print("")
    print("")
    print("Calculating K':")

    #second algorithm
    H = Cross_leverage_scores(A)


    #########################################################################


    #uncomment the test you want to run (notice that the first one loads
    # it's data set and the other two need a data set from the main)

    ############Tests for algorithm 1:############

    # calculate_average_statistical_score_per_size()
    # calculate_average_statistical_score_per_epsilon(df)

    ############Tests for algorithm 2:############

    #calculate_heavy_pairs_per_epsilon(A)

if __name__=="__main__":
    main()

