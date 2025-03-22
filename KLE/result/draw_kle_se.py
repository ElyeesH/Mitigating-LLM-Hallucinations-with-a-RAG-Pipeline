import pickle
import matplotlib.pyplot as plt
import numpy as np
file_path = "test/squad/uncertainty_measures.pkl"
file_path1="test/squad/validation_generations.pkl"
with open(file_path, "rb") as f:
        data = pickle.load(f)
with open(file_path1, "rb") as f1:
        data1 = pickle.load(f1)
        
#KE=np.zeros((1,200))
#se=np.zeros((1,200))
KE=[]
se=[]
expriments=400 
for ex in range(400):
    question=data1[list(data1.keys())[ex]]["question"] 
    acc=data1[list(data1.keys())[ex]]["most_likely_answer"]["accuracy"]
    if acc==0: 
        
        SE=data["uncertainty_measures"]["semantic_entropy"][ex]
        diff=[]
        
        
        KLE=data["uncertainty_measures"]["semantic_kernel_heat_t=0.2_alpha_1.0"][ex]
        
        KE.append(KLE)
        se.append(SE)
        

            

    

KE=np.array(KE)
se=np.array(se)
print(np.mean(KE==se))
plt.figure(figsize=(8, 6))

plt.hist(KE, bins=50, alpha=0.4, color='blue', edgecolor='black', linewidth=1.2, label='Kernel Language Entropy')
plt.hist(se, bins=50, alpha=0.4, color='orange', edgecolor='black', linewidth=1.2, label='Semantic Entropy')

plt.title('Distribution of KLE and SE Variables', fontsize=14, fontweight='bold')
plt.xlabel('Values of KLE and SE', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()