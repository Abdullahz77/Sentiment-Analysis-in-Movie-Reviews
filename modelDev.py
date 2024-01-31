import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os

# Defining a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
def createOptimizer(model,learningRate):
    optimizer = torch.optim.AdamW(model.parameters(), learningRate)
    return optimizer

# Defining the training function
def train(model,tokenizer, train_dataloader, val_dataloader, optimizer, epochs,path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                predictions = torch.argmax(outputs.logits, dim=1)
                total += labels.size(0)
                val_loss += loss.item()
                correct += (predictions == labels).sum().item()

            average_val_loss = val_loss / len(val_dataloader)
            accuracy = correct / total
    pM=os.path.join(path,'modelA')
    pT=os.path.join(path,'tokenizerA')
    os.makedirs(pM,exist_ok=True)
    os.makedirs(pT,exist_ok=True)
    model.save_pretrained(pM)
    tokenizer.save_pretrained(pT)
    return average_val_loss,accuracy

#Defining the evaluation function
def evaluate(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss

            total_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    average_loss = total_loss / len(test_dataloader)
    accuracy = total_correct / total_samples
    return average_loss,accuracy

#Defining the predict function
def modelPredictions(model,input,tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    tokenizedInput=tokenizer.encode_plus(
        input,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt')
    inputIds = tokenizedInput['input_ids'].to(device)
    attentionMask = tokenizedInput['attention_mask'].to(device)
    with torch.no_grad():
        outputs=model(attention_mask=attentionMask,input_ids=inputIds)
    probabilities = F.softmax(outputs.logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    # Decode the predicted labels
    label_mapping = {0: 'Negative', 1: 'Positive'} 
    decoded_predictions = [label_mapping[p.item()] for p in predictions]
    return decoded_predictions[0]

def createTrainingData(trainTexts, trainLabels, trainBatchSize,tokenizer):
    trainDataset=CustomDataset(trainTexts, trainLabels, tokenizer,max_length=128)
    trainDataLoader=DataLoader(trainDataset, batch_size=trainBatchSize, shuffle=True)
    return trainDataLoader

def createValidationData(valTexts,valLabels,valBatchSize,tokenizer):
    valDataset=CustomDataset(valTexts,valLabels,tokenizer,max_length=128)
    valDataLoader=DataLoader(valDataset,batch_size=valBatchSize,shuffle=False)
    return valDataLoader

