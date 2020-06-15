import pandas as pd

def Append_Sequence(listt , value , length):
    for i in range(0,length):
        listt[length-1-i] = listt[length-1-i-1]
    listt[0] = value
    return listt



data = pd.read_csv('driving_log.csv')

counter = int(input("Enter Size of your Dataset :"))
Speed_Sequence = [0,0,0,0,0,0,0,0,0,0]
data2=[]
# read row line by line
for d in data.values:
    #if d[6]=='nan':
    Speed_Sequence = Append_Sequence(Speed_Sequence , d[7] , 10)

    data2.append([str(Speed_Sequence)])
    #print("missed")
    #print (d[6])
    #break
#else:
    #counter=counter+1  
#for d in data.values:
#    for i in range(0,8):
#        print(d[i])
df = pd.DataFrame(data2, columns = ['Speed_Sequence'])  
df.to_csv('files_path.csv', header=None)
