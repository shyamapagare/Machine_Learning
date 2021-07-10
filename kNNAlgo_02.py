import numpy
import statistics
import matplotlib.pyplot
import math
from mpl_toolkits import mplot3d

class KNNClassifier():

	plot_data=False
	classifier_type='Classification'
	colors_and_markers=('r','o')
	query_color='k'
	query_marker='o'
	query_marker_size=20
	
	def __init__(self,**kwargs):
		if 'dataset' in kwargs.keys():
			self.dataset=kwargs['dataset']
		if 'query_dataset' in kwargs.keys(): self.query_dataset=kwargs['query_dataset']
		if 'classifier_type' in kwargs.keys(): 
			self.classifier_type=kwargs['classifier_type']
		else:
			self.classifier_type='Classification'
		if 'plot_data' in kwargs.keys():
			if kwargs['plot_data']==True:
				if 'colors_and_markers' in kwargs.keys(): self.colors_and_markers=kwargs['colors_and_markers']
				if 'query_color' in kwargs.keys(): self.query_color=kwargs['query_color']
				if 'query_marker' in kwargs.keys():self.query_marker=kwargs['query_marker']
				if 'query_marker_size' in kwargs.keys():self.query_marker_size=kwargs['query_marker_size']
	
	def process(self):
		distancesList=[]
		ep=len(self.query_dataset)

		for p in self.dataset:
			qi=0
			sumOfSquares=0	
			while qi<ep:
				qd=(abs(float(p[qi])-float(self.query_dataset[qi])))**2
				sumOfSquares=sumOfSquares+qd
				qi+=1
			distancesList.append((p[qi],math.sqrt(sumOfSquares)))
		distancesList.sort(key=lambda x:x[1])
		kFactor=int(math.sqrt(len(distancesList)))
		if kFactor%(len(distancesList))==0 : kFactor+=1
		if kFactor%2==0 : kFactor=kFactor+1
		topKElements=distancesList[:kFactor]
		topKClass=[p[0] for p in topKElements]
		if self.classifier_type=='Regression' or self.classifier_type=='regression' :
			result=statistics.median(topKClass)
		else :
			result=statistics.mode(topKClass)
		return result

my_data_set=numpy.array([[33.1,110.3,67,20,56,'Sunny'],[21.4,265.9,61,15,40,'Sunny'],[45.4,142.7,50,10,22,'Sunny'],[36.6,260.9,69,35,20,'Sunny'],	[39.6,259.5,56,40,35,'Sunny'],[53.3,168.3,61,54,40,'Sunny'],[37.1,160.8,57,44,15,'Sunny'],[36.0,192.1,59,26,40,'Sunny'],[23.4,412.9,68,30,115,'Rainy'],[21.2,568.0,69,35,141,'Rainy'],[20.0,674.9,57,45,85,'Rainy' ],[16.5,711.8,56,10,90,'Rainy' ],[21.9,523.9,56,20,75,'Rainy' ],[18.2,659.3,57,65,115,'Rainy'],[16.0,753.9,66,40,140,'Rainy'],[22.6,713.7,60,30,100,'Rainy'],[15.1,746.7,63,75,85,'Rainy'],[-0.2,949.5,59,5,10,'Cold'],[5.4,944.9,61,20,8,'Cold'],[-2.9,1008.1,65,15,15,'Cold'],[3.8,1080.8,61,20,20,'Cold'],[4.5,787.9,69,10,9,'Cold'],[-3.6,941.9,60,10,6,'Cold'],[2.6,1040.8,61,5,10,'Cold'],[3.3,1183.4,64,15,15,'Cold'],[7.0,1190.0,67,25,20,'Cold'],[8.4,970.4,57,20,20,'Cold']])		

my_query_dataset=[17.0,780.3,57,65,80]

my_classifier=KNNClassifier(dataset=my_data_set,query_dataset=my_query_dataset,classifier_type='Classification',plot_data=False)
result=my_classifier.process()
print('Data points : {} belongs to class :{}'.format(my_query_dataset,result))

new_dataset=numpy.array([[10,20,'A'],[20,25,'A'],[15,20,'A'],[30,45,'A'],[10,10,'A'],[25,20,'A'],[30,55,'A'],[40,60,'A'],[90,100,'A'],[80,60,'A'],[110,120,'B'],[120,125,'B'],[115,120,'B'],[130,145,'B'],[110,110,'B'],[125,120,'B'],[130,155,'B'],[140,160,'B'],[80,120,'B']])

new_query=[80,100]
my_classifier=KNNClassifier(dataset=new_dataset,query_dataset=new_query,classifier_type='Classification',plot_data=False)
result=my_classifier.process()
print('Data points : {} belongs to class :{}'.format(new_query,result))
