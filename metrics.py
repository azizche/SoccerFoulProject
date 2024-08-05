import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

class ConfusionMatrix:
    def __init__(self,class_names:list[str]):
        self.num_classes=len(class_names)
        self.matrix= np.zeros((self.num_classes,self.num_classes))
        self.class_names=class_names

    def process(self, predictions,labels):
        for pred_class,label_class in zip(predictions,labels):
            self.matrix[label_class][pred_class]+=1
    
    def compute_accuracy(self,eps=1e-7):
        return np.sum(self.matrix.trace())/(np.sum(self.matrix)+eps)
    
    def plot(self,normalized=False,show=False,save=False,path='',prefix=''):
        fig=plt.figure(figsize=(12,10))
        title='Confusion Matrix'
        matrix=self.matrix
        if normalized:
            matrix=matrix.astype('float') /np.max(matrix,axis=0)
            title= 'Normalized '+title
        sns.heatmap(matrix,xticklabels=self.class_names,yticklabels=self.class_names,cmap="Blues", linewidths=.5,fmt='.2f' if normalized else ".2g",annot=True)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(title)
        if show:
            plt.show()
        if save:
            normalized_str='Normalized' if normalized else ''
            fig.savefig(path.__str__()+f'/{prefix}_Confusion_Matrix_{normalized_str}.png')


