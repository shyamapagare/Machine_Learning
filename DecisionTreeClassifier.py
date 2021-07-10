import numpy
import math
import statistics
from collections import Counter

training_dataset=numpy.array([[True,True,True,True],[True,True,False,False],[True,False,False,False],[True,False,False,True],[False,False,True,True],[False,True,True,False],[False,False,False,True],[False,False,False,False],[True,False,True,True],[True,True,False,True],[False,True,False,False],[False,True,True,True]])

test_dataset=[True,False,False]


def calculate_entropy(dataset):
	dataset_T=dataset.T
	target_indx=dataset.shape[1] #4
	target_data_sz=dataset.shape[0] #12

	base_en_cntr=Counter(dataset_T[target_indx-1])
	base_entrpy=0
	for e in base_en_cntr:
		base_entrpy=base_entrpy+(float(base_en_cntr[e])/target_data_sz)*math.log((float(base_en_cntr[e])/target_data_sz),2)
	base_entrpy=-1*base_entrpy
	
	IG_features={}
	for fi in range(target_indx-1):
		f_en_cntr=Counter(dataset_T[fi]).most_common()
		f_en_data={}
		for data in dataset:
			if (data[fi],data[target_indx-1]) in f_en_data.keys():
				val=f_en_data[(data[fi],data[target_indx-1])]+1
			else :
				val=1
			f_en_data[(data[fi],data[target_indx-1])]=val
		w_f_ls={}
		total_f_entopy=0
		for t in f_en_cntr:
			x=0
			for k,v in f_en_data.items():
				if t[0]==k[0]:
					x-=((float(v)/t[1])*math.log(float(v)/t[1],2))
			w_f_ls[t[0]]=t[1]/target_data_sz*x
			total_f_entopy+=t[1]/target_data_sz*x
		IG_fi=base_entrpy-total_f_entopy
		IG_features[fi]=IG_fi
	max_ig = max(IG_features, key=IG_features.get)
#	print('MAx IG :',max_ig)
#	print('####Returning IG#####')
	return max_ig
	
def decision_tree_calc(training_dataset):
	result_dict={}

	training_dataset_T=training_dataset.T
	num_of_unqiue_target=numpy.unique(training_dataset_T[training_dataset[0].size-1],0)
	if num_of_unqiue_target.size ==1:
		target_class=num_of_unqiue_target[0]
		result_dict['target_class']=target_class
	elif training_dataset.shape[1]==2:
		target_class=statistics.mode(training_dataset_T[1])
		result_dict['target_class']=target_class
	else :
		split_by_feature_idnx=calculate_entropy(training_dataset)
		splited_data_dict={}
		for i in range(training_dataset.shape[0]):
			if training_dataset[i][split_by_feature_idnx] in splited_data_dict.keys():
				splited_data_dict[training_dataset[i][split_by_feature_idnx]]=numpy.vstack((splited_data_dict[training_dataset[i][split_by_feature_idnx]],[training_dataset[i]]))
			else:
				splited_data_dict[training_dataset[i][split_by_feature_idnx]]=numpy.array([training_dataset[i]])
		result_dict['split_by_feature_idnx']=split_by_feature_idnx
		result_dict['splited_data_dict']=splited_data_dict
	return result_dict
		

print(training_dataset)
print('*'*20)	
print(test_dataset)
print('*'*20)
result_dict={}
feature_indx=0
splited_data_dict={}
test_feature_val=None

while 'target_class' not in result_dict.keys():
	if 'split_by_feature_idnx' in result_dict.keys():
		feature_indx=result_dict['split_by_feature_idnx']
		splited_data_dict=result_dict['splited_data_dict']
		test_feature_val=test_dataset[feature_indx]
		if test_feature_val in splited_data_dict.keys():
			new_training_dataset=splited_data_dict[test_feature_val]
			result_dict=decision_tree_calc(numpy.delete(new_training_dataset,result_dict['split_by_feature_idnx'],1))
	else:
		result_dict=decision_tree_calc(training_dataset)

print(result_dict['target_class'])