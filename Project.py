from modelDev import *
from preprocessing import *
import sys
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd

def trainPipeline(data,customModel,customTokenizer,batchSize,lr=5e-5,trainEpochs=5,exportPath='custom',augment=False):
    '''This function trains a desired BERT model on the reviews data.
    !!!Ensure that the data supplied is a dataframe for the pipeline to work as intended'''
    #Preprocessing the data
    data=process(data)

    #loading the data into a list
    reviews=[x for x in data['review']]
    sentiments=[x for x in data['sentiment']]

    #Splitting the data into train,validation and test splits
    trainText,testText,trainLabels,testLabels=train_test_split(reviews,sentiments,test_size=0.2,shuffle=True,random_state=42)
    trainningText,valText,trainningLabels,valLabels=train_test_split(trainText,trainLabels,test_size=0.1,shuffle=True,random_state=42)

    #Augmenting the data if the user specifies to do so
    if augment:
        aug=createAugmenter()
        augmentedText,augmentedLabels=augmentText(trainningText,trainningLabels,aug)
        for t,s in zip(augmentedText,augmentedLabels):
            trainningText.append(t)
            trainningLabels.append(s)
    #Creating data loaders for the model
    trainnningData=createTrainingData(trainningText,trainningLabels,batchSize,customTokenizer)
    valDataset=createValidationData(valText,valLabels,batchSize,customTokenizer)
    testDataset=createValidationData(testText,testLabels,batchSize,customTokenizer)

    #Creating the optimizer
    optim=createOptimizer(customModel,lr)


    #trainning and evaluating the model
    valLoss,valAccuracy=train(
        train_dataloader=trainnningData,
        val_dataloader=valDataset,
        model=customModel,
        tokenizer=customTokenizer,
        optimizer=optim,
        path=exportPath,
        epochs=trainEpochs
    )

    testLoss,testAccuracy=evaluate(
        test_dataloader=testDataset,
        model=customModel
    )
    
    with open('runs.txt','a') as file:
        file.write(f'epochs = {trainEpochs} Learning Rate = {lr}\nValidation Loss = {valLoss} Validation Accuracy = {valAccuracy}\nTesting Loss = {testLoss} Testing Accuracy = {testAccuracy}\n********\n')


def predictSentiment(data,customModel,customTokenizer,outputPath):
    '''This function runs the model on reviews and returns the sentiment class for each review.
    !!!Ensure that the reviews are in a list for the pipeline to work as intended'''
    #Preprocessing the data
    data=pd.Series(data)
    data=data.apply(lambda x: removeTags(x))
    data=data.apply(lambda x: removeURL(x))
    data=data.apply(lambda x: removeEmoji(x))
    data=data.apply(lambda x: removeBetweenBrackets(x))
    sentences=[]
    revID=[]
    sentiments=[]
    id=1
    for i in data:
        for sents in splitToSent(i):
            sentiment=modelPredictions(
                model=customModel,
                input=sents,
                tokenizer=customTokenizer
            )
            sentiments.append(sentiment)
            sentences.append(sents)
            revID.append(id)
        id+=1
    df={
        'Review Segment':sentences,
        'Sentiment':sentiments,
        'Original Review ID':revID
    }
    df=pd.DataFrame(df)
    df.to_csv(outputPath,index=False)

    

if __name__=='__main__':
    MODEL_PATH=r"C:\Users\User\Downloads\Project\Project\modelsAugmented\bert22"
    TOKENIZER_PATH=r"C:\Users\User\Downloads\Project\Project\tokenizersAugmented\bert22"
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    model=BertForSequenceClassification.from_pretrained(MODEL_PATH)
    if len(sys.argv)>1:
        args=sys.argv[1:]
        mode=args[0]
        dataPath=args[1]
        if dataPath.endswith('.csv'):
            dataset=True
            df=pd.read_csv(dataPath)
        else:
            df=dataPath.split(',,')

        if (mode=='train' and dataset):
            batchS=args[2]
            if (len(args)>3 and len(args)<=7):
                learningRate=float(args[3])
                numEpochs=int(args[4])
                modelExportPath=args[5]
                if args[6].lower()=='true':
                    aug=True
                else:
                    aument=False
            trainPipeline(
                customModel=model,
                customTokenizer=tokenizer,
                batchSize=batchS,
                exportPath=modelExportPath,
                trainEpochs=numEpochs,
                lr=learningRate,
                augment=aug,
                data=df
            )
        elif (mode=='pred' and len(args)<4):
            predictionsExportPath=args[3]
            predictSentiment(
                customModel=model,
                customTokenizer=tokenizer,
                data=df
            )
        else:
            print('PLease enter valid arguments')
        
    text=['In night scenes you can\'t see anything','You know a lot people say this about a ton of different shows but Game of Thrones is absolutely, without a doubt, one of the best TV shows ever created...it\'s damn near perfect (except for maybe the last few episodes of the last season) and easily one of my all-time favorite shows! Yeah, the last few episodes of Season 8 weren\'t that good and the ending was just awful but the first 7 1/2 seasons were so amazing that it still gets a 10 from me! It\'s one of the rare shows where I can watch it all the way from the first episode to the last and never get bored. I\'m pretty sure there\'s never been a show more talked about or more loved in the history of television than GOT. All you have to do is read through the reviews to see how loved this show really is! It\'s absolutely terrible.']
    df=pd.read_csv('IMDBSEnt.csv')
    predictSentiment(
        customModel=model,
        customTokenizer=tokenizer,
        data=text,
        outputPath='preds.csv'
    )
    trainPipeline(
        data=df,
        customModel=model,
        customTokenizer=tokenizer,
        batchSize=8,
        exportPath='new/'
    )
