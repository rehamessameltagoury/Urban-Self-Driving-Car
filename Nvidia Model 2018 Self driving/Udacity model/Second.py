import pandas as pd

def Append_Sequence(listt , value , length):
    for i in range(0,length):
        listt[length-1-i] = listt[length-1-i-1]
    listt[0] = value
    return listt

def str_to_float(inp):
    inp = inp[1:-1]
    res = [float(idx) for idx in inp.split(',')]
    return res

data = pd.read_csv('files_path.csv')

counter = int(input("Enter Size of your Dataset :"))
data2=[]
# read row line by line
i = 0
for d in data.values:
    classes = [ 0,0,0]
    current_speed = str_to_float (d[1])
    #Speed_Sequence = Append_Sequence(Speed_Sequence , d[7] , 10)
    if ((current_speed[0] - current_speed[9])* 0.44704 )> 0.25:
        classes = [ 0,0,1]
        data2.append(str(classes))
        #print(current_speed[0])
        #print(int(current_speed[0]))
    elif ((current_speed[0] - current_speed[9])* 0.44704 )< -0.25:
        classes = [ 1,0,0]
        data2.append(str(classes))
    else:
        classes = [ 0,1,0]
        data2.append(str(classes))
    i+=1
    if i >= counter:
        break
 
df = pd.DataFrame(data2, columns = ['Speed_Classes'])  
df.to_csv('test.csv', header=None)