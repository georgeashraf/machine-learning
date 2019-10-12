from PIL import Image
import string
import matplotlib.pyplot as plt
import os
import math
def read_data(mode):
    data=[]
    labels=[]
    if mode is "Train":
      directory="Train/"
    elif mode is "Test" :
      directory="Test/"
    for filename in os.listdir(directory):
        image = Image.open(str(directory)+str(filename))
        pix_val = list(image.getdata())
        normalized_pixels = [x / 255 for x in pix_val]
        data.append(normalized_pixels)
        x = filename.find(".")
        labels.append(ord(filename[x-2])-96)
    return data,labels    

def mean(L):
   return sum(L)/float(len(L))


def stdev(L):
    avg = mean(L)
    variance = sum([pow(x-avg,2) for x in L])/float(len(L)-1)
    return math.sqrt(variance)


def learn_parameters(data,labels):
    """
    learn mean and stdv for each attribute for each class
    """
    count=1
    indx=0
    data_single_class=[]
    model_params=[]
    for i in range(0,26):
        while count<8:
           data_single_class.append(data[indx])
           count=count+1
           indx=indx+1
        params_of_one_class = [(mean(attribute), stdev(attribute)) for attribute in zip(*data_single_class)]
        model_params.append(params_of_one_class)
        count=1
        data_single_class=[]
    return model_params

def calculateProbability(x, mean, stdev):
    if(stdev<0.1 ):
        stdev=0.1
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent


def predict(data_instance,model_params):
    """ multiplication of all attributes for each class and choosing class with highest prob

    """
    probabilities=[]
    for alphabet in model_params:   
        prob=1
        for indx,value in enumerate(data_instance):
            l=alphabet[indx]
            feature_prob=calculateProbability(value,l[0],l[1])
            #feature_prob=feature_prob*100
            if feature_prob <0.1:
                feature_prob=0.1

            prob*=feature_prob
            
        probabilities.append(prob)
    return probabilities.index (max(probabilities))+1

def test(test_data,model_params):
    predictions=[]
    for i in test_data:
        x=predict(i,model_params)
        predictions.append(x)
    return predictions    

if __name__ == '__main__':
    data,labels=read_data("Train")
    test_data,test_labels=read_data("Test")
    #print(test_labels)
    model_params=learn_parameters(data,labels)
    predictions=test(test_data,model_params)
   # print(predictions)
    accuracy=[]
    i=0
    while(i<(26*2)):
        score=0
        if(predictions[i]==test_labels[i]):
            score+=1
        if(predictions[i+1]==test_labels[i+1]):
            score+=1
        accuracy.append(score)
        i+=2
    alphabet=list(string.ascii_lowercase)
    plt.title("Accuracy") 
    plt.xlabel("Alphabet") 
    plt.ylabel("correctly classified images") 
    plt.plot(alphabet,accuracy,'ro') 
    plt.savefig('accuracy.png')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    